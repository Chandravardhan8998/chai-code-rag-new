from typing import List, TypedDict

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.constants import START,END
from langgraph.graph import StateGraph
import requests
from bs4 import BeautifulSoup
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
import os
import asyncio
import json

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionAssistantMessageParam
from langchain.schema import Document
from starlette.responses import StreamingResponse

chai_doc_collection="chai_doc_documents"
from qdrant_client import QdrantClient, models

class HrefData(BaseModel):
    href: str
    title: str

class SplitTextData(TypedDict):
    split_text:List[Document]
    collection_name:str

class Course(TypedDict):
    name: str
    paths: List[str]
class State(TypedDict):
    prompt:str
    response:List
    selected_topics:None|List[Course]
    split_text_data:None|List[SplitTextData]
    ui_response:str

load_dotenv()
qdb_api = os.getenv("QDRANT_DB_API")
qdb_url = os.getenv("QDRANT_DB_URL")
rag_collection="learning_vectors"

async def manage_collection_building(collection_name:str):
    q_client = QdrantClient(url=qdb_url, api_key=qdb_api)
    is_exist = q_client.collection_exists(rag_collection)
    if not is_exist:
        try:
            q_client.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE), )
            return True
        except:
            print("failed to create database")
            return False
    return True

async def get_html(path:str):
    # url = 'https://chaidocs.vercel.app/youtube/getting-started/'
    url = path

    response = requests.get(url)
    if response.status_code != 200:
        print(url)
        raise Exception('Network response was not ok')

    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    return soup

async def get_paths():
    soup=await get_html('https://chaidocs.vercel.app/youtube/getting-started/')
    elements = soup.find_all(class_='top-level')

    titles = soup.find_all(class_='large astro-3ii7xxms')
    blog_titles = [el.get_text(strip=True) for el in titles]

    hrefs = []
    for element in elements:
        links = element.find_all('a')
        for index,link in enumerate(links):
            href = link.get('href')
            text = link.get_text()
            if href:
                href_data = HrefData(href=href.replace("youtube/", ""), title=text)
                hrefs.append(href_data)
    result = {}
    counter=0
    for index,link in enumerate(hrefs):
        parts = link.href.strip("/").split("/")
        if len(parts) == 1:
            section = parts[0]
            if section not in result:
                result[section] = {
                    "name": section,
                    "paths": ["/"],
                    "title": blog_titles[counter],
                    "paths_title":[link.title],
                }
                counter=counter+1
        elif len(parts) > 1:
            section = parts[0]
            sub_path = "/".join(parts[1:])
            if section not in result:
                result[section] = {
                    "name": section,
                    "paths": [],
                    "paths_title":[],
                    "title":blog_titles[counter]
                }
                counter = counter + 1
            result[section]["paths"].append(sub_path)
            result[section]["paths_title"].append(link.title)
    # print("====================")
    # print(result)
    # print("====================")
    return result


class TopicSlug(BaseModel):
    parent_slug:str
    sub_topic_slugs:List[str]

class ExtractionResponse(BaseModel):
    slugs:List[TopicSlug]
    message:str

class GetBlogSubContent(BaseModel):
    topic: str
    sub_topic:list[str]
class ExtractionSubtopicsResponse(BaseModel):
    slugs:List[GetBlogSubContent]
    message:str



async def extract_topic(state:State):
    # print("🤖 --- doing write_code",state.get("node_data"))
    load_dotenv()
    client = OpenAI()
    # get list of topics
    topics=await get_paths()
    titles=[]
    #  Define prompts
    for key in topics.keys():
        topic=topics.get(key)
        sub_topics=[]
        for index,subtopic_path in enumerate(topic.get("paths")):
            sub_topic=f"""<sub-topic>
                <sub-topic-heading>{topic.get("paths_title")[index]}</sub-topic-heading>
                <sub-topic-slug>{subtopic_path}</sub-topic-slug>
                </sub-topic>"""
            sub_topics.append(sub_topic)
        title_meta=f"""-<topic>
                        <title>{topic.get("title")}</title> 
                        <slug>{topic.get("name")}</slug>
                        <sub-topics>
                            {"\n".join(sub_topics)}
                        </sub-topics>
                        </topic>  
                            """
        titles.append(title_meta)
    system_prompt = f"""<instruction>
    You are a data classification expert. Your job is to classify user prompts into valid topic slugs and their related sub-topic slugs.

    ⚠️ This is part of our business logic, so your output must be **100% accurate and deterministic**.

    Your task:
    - Go through all topics and match the user query against all provided topics and their sub-topic-headings.
    - Include topics that matches if  the user prompt includes multiple valid areas and leave topics that are not present in available topics .
    - if any one of two topics in prompt available in topics return that one slug only and leave rest data on user.
    - For each match, map **only the sub-topic-slugs that belong to their correct parent_slug**.

    Rules:
    1. Only use the topics and sub-topic-headings defined in the list below.
    2. For each valid match:
       - Return the corresponding `parent_slug`
       - Include only the relevant `sub_topic_slugs` that belong to that parent_slug
    3. If the user query includes a mix of valid and invalid topics, include only the valid ones. Ignore the rest.
    4. If the query includes multiple valid topics, include all of them in the response.
    5. Use `<sub-topic-heading>` to find matching sub-topic-slugs. Do not guess or infer outside this list.
    6. Do NOT include sub-topic-slugs from one parent_slug under another.
    7. Output format must strictly follow this Python class structure:
    
    
    
    ```python
    class TopicSlug(BaseModel):
        parent_slug: str
        sub_topic_slugs: List[str]

    class ExtractionResponse(BaseModel):
        slugs: List[TopicSlug]  # This is a List
        message: str
        ```
Available Topics:
<topics> {"\n".join(titles)} </topics> </instruction>
<example>
Prompt: how to add SQL in python
1: Thinking: DO we have SQL in our topics `Yes`
2: Thinking: DO we have Python in our topics `No`
3: I should return SQL related data and slugs and cna leave python upto use as i dont hvae that in topics
</example>
"""
    system_prompt_chat = ChatCompletionSystemMessageParam(role="system", content=system_prompt),

    response:List=state.get("response")
    messages=[
        *response,
    ]
    is_untouched_prompt=len(response)==1
    # if tool_chat and not is_untouched_prompt:
    #     messages.append(tool_chat)
    query_res = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
    response_format=ExtractionResponse,
        messages=[
            *system_prompt_chat,
            *messages]
    )
    parsed_response = query_res.choices[0].message.parsed
    print("SELECTED TOPICS: ",parsed_response)
    response_chat_data=ChatCompletionAssistantMessageParam(role="assistant",content=parsed_response.message)
    messages.append(response_chat_data)

    selected_topics:List[Course]=[]
    for topic_subject in parsed_response.slugs:
        # paths:List[str]=topics.get(topic_subject.parent_slug).get("paths")
        paths:List[str]=topic_subject.sub_topic_slugs
        course:Course={
               "name": topic_subject.parent_slug,
            "paths":paths
        }
        selected_topics.append(course)

    state["response"] = messages
    state["selected_topics"] = selected_topics
    ui_response = {
        "current_step":"extract_topic",
        "ui_response_text": parsed_response.message,
        "next_step": "extract_sub_topic"
    }
    state["ui_response"] = json.dumps(ui_response)
    return state
async def enhance_prompt(state:State):
    # print("🤖 --- doing write_code",state.get("node_data"))
    load_dotenv()
    client = OpenAI()
    # get list of topics
    topics=await get_paths()
    print(topics)
    titles=[]
    #  Define prompts
    for key in topics.keys():
        topic=topics.get(key)
        title_meta=f"title:{topic.get("title")}, slug:{topic.get("name")}"
        titles.append(title_meta)

    system_prompt = f"""<instruction>
    You are a proofreader and prompt enhancer. Your role is to take a rough user prompt and refine it by adding necessary context so that a Language Model (LLM) can understand and respond effectively.

    You are restricted to working ONLY with the following approved topics. You MUST NOT enhance or respond based on any content outside these topics. Do not assume or invent extra context.

    <topics>
    -{"\n-".join(titles)}
    </topics>

    Your task:
    1. Check if the user's input is related to any of the topics listed above. Use both the topic title and slug for matching.
    2. If the input is related to a topic, enhance the prompt to add clarity and context — but stay strictly within that topic.
    3. If the input is NOT related to any topic, return exactly: **"out of context info"** (do not include quotes).
    
    Do not add, assume, or create anything outside the listed topics.
</instruction>
"""
    print(system_prompt)
    # Add Chat item
    system_prompt_chat = ChatCompletionSystemMessageParam(role="system", content=system_prompt),

    response:List=state.get("response")
    messages=[
        *response,
    ]

    query_res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            *system_prompt_chat,
            *messages]
    )
    content = query_res.choices[0].message.content



    response_chat_data=ChatCompletionAssistantMessageParam(role="assistant",content=content)
    messages.append(response_chat_data)

    state["response"] = messages
    print("messages ---------------------------")
    print(content)
    ui_response = {
        "current_step":"enhance_prompt",
        "ui_response_text":content,
        "next_step": "extract_topic"
    }
    state["ui_response"] = json.dumps(ui_response)
    return state

async def extract_sub_topic(state:State):
    load_dotenv()
    client = OpenAI()
    # get list of topics
    topics=state.get("selected_topics")
    topics_prompt=""
    for topic in topics:
        print(topic.get("paths"))
        sub_topics = "\n".join([f"<slug> {path} </slug>" for path in topic.get("paths", [])])
        topics_prompt+=f"""
        <topic>
        <topic-name>
        {topic.get("name")}
        </topic-name>
        <sub-topics>
            <slugs>
                {sub_topics}
            <slugs>
        </sub-topics>
        <topic>
        """
        print("WRITING topics_prompt")
    print(topics_prompt)
    #  Define prompts
    system_prompt = f"""
                 <instruction>
               You are an data classification expert and you are expert at classifying and extracting
               matching sub-topics based on provided prompt
               Check if provided sub-topics directly make any connection,
               this sub-topics could be multiple 
               output sub-topics should from this list only if found any return slugs (<slug>)
               we will use this slug (<slug>) to generate URL's later so this data is precious to us.
               Do not generate any data out of context of provided sub-topics
        </instruction>
        <example>
            <slug>some-slug-1</slug>
            <slug>some-slug-2</slug>
            <slug>some-slug-3</slug>
            Expected: return should be only this ["some-slug-1","some-slug-2"] (selection would be based on prompt)
            Unexpected or Error: ["Some slug","Slug","Data"] this is going out of context
        </example>
        <strict_instruction>
        if prompt looks like this "out of context info" or no related data found you will return nothing.
        </strict_instruction>
           <content>
           {topics_prompt}
           </content>
        """

    # Add Chat item
    # tool_chat = ChatCompletionUserMessageParam(role="user", content=state.get("prompt"))
    system_prompt_chat = ChatCompletionSystemMessageParam(role="system", content=system_prompt),

    response:List=state.get("response")
    messages=[
        *response,
    ]
    is_touched_prompt=len(response)==1
    # if tool_chat and not is_touched_prompt:
    #     messages.append(tool_chat)
    query_res = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
    response_format=ExtractionSubtopicsResponse,
        messages=[
            *system_prompt_chat,
            *messages]
    )
    parsed_response = query_res.choices[0].message.parsed


    response_chat_data=ChatCompletionAssistantMessageParam(role="assistant",content=parsed_response.message)
    messages.append(response_chat_data)
    selected_topics=[]
    for topic_subject in parsed_response.slugs:
        topic_info={
            "name":topic_subject.topic,
            "paths":topic_subject.sub_topic,
        }
        selected_topics.append(topic_info)
    state["response"] = messages
    state["selected_topics"] = selected_topics

    ui_response = {
        "current_step": "extract_sub_topic",
        "ui_response_text": parsed_response.message,
        "next_step": "split_text_doc"
    }
    state["ui_response"] = json.dumps(ui_response)
    return state

async def get_content(path:str):
    url = f"https://chaidocs.vercel.app/youtube/{path}"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: BeautifulSoup(x, "html.parser").text
    )
    docs = loader.load()

    # Sort the list based on the URLs and get the text
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    return concatenated_content


async def get_split_text(course:Course)->SplitTextData:

    topic=course.get("name")
    sub_topics=course.get("paths")

    # split text doc
    blog_data: list[dict[str, str]] = []
    print("====> ",sub_topics," <=====")
    if len(sub_topics)==0:
        path=f"{topic}"
        text=await get_content(path)
        blog_data.append({"text":f"blog-path:{path} content: {text}","path":f"https://chaidocs.vercel.app/youtube/{path}"})
    else:
        for sub_topic in sub_topics:
            path=f"{topic}/{sub_topic}"
            text=await get_content(path)
            blog_data.append({"text":f"blog-path:{path} blog-subject:{sub_topic} content: {text}","path":f"https://chaidocs.vercel.app/youtube/{path}"})

    split_text=[]
    for blog in blog_data:
        split_text.append(Document(page_content=blog.get("text"),metadata={"source": blog.get("path")}))
    if len(split_text)==0:
        response: SplitTextData = {
            "collection_name": "",
            "split_text": []
        }
        return response
    # split tet doc end

    # manage collection start

    collection_name=f"{chai_doc_collection}_{topic}"
    collection_exist= await manage_collection_building(collection_name)
    print("collection_exist ",collection_exist)
    if not collection_exist:
        response: SplitTextData = {
            "collection_name": "",
            "split_text": []
        }
        return response
    response:SplitTextData={
"collection_name":collection_name,
        "split_text":split_text
    }
    return response
    # manage collection end
    # await embedd_split_text(split_text=split_text,collection_name=collection_name)


    # embed start
async def embedd_split_text(split_text:List[Document],collection_name:str):
    vector_db = embedd_vector_db(split_text,collection_name=collection_name)
    if not vector_db:
        return {"isSuccess":False,"message":"Embedding failed."}
    # embed end
    return {"isSuccess": True, "message": "Embedded successfully."}


def embedd_vector_db(split_text: list[Document],collection_name:str) -> bool | QdrantVectorStore:
    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        vector_db = QdrantVectorStore.from_documents(
            documents=split_text,
            url=qdb_url,
            collection_name=collection_name,
            embedding=embedding_model,
            force_recreate=True,
            api_key=qdb_api
        )
        return vector_db
    except Exception as e:
        print(e)
        return False

async def split_text_doc(state:State):
    load_dotenv()
    split_text_data:List[SplitTextData]=[]
    for blog in state.get("selected_topics"):
        response:SplitTextData=await get_split_text(blog)
        if response.get("collection_name"):
            split_text_data.append(response)
    state["split_text_data"]=split_text_data
    ui_response = {
        "current_step": "split_text_doc",
        "ui_response_text": "Text Splitting done.",
        "next_step": "embedd_topic"
    }
    state["ui_response"] = json.dumps(ui_response)
    return state



async def embedd_topic(state:State):
    split_text_data=state.get("split_text_data")
    for data in split_text_data:
       response= await embedd_split_text(split_text=data.get("split_text"),collection_name=data.get("collection_name"))
    ui_response={
        "current_step": "embedd_topic",
        "ui_response_text":"Embedding successful",
        "next_step":"run_user_query"
    }
    state["ui_response"]=json.dumps(ui_response)
    return state

async def run_user_query(state:State):
        client = OpenAI()

        # Vector Embeddings
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        selected_topics=state.get("selected_topics")
        prompt=state.get("prompt")
        lines = []
        for topic in selected_topics:
            collection_name = f"{chai_doc_collection}_{topic.get("name")}"
            vector_db = QdrantVectorStore.from_existing_collection(
                url=qdb_url,
                api_key=qdb_api,
                collection_name=collection_name,
                embedding=embedding_model
            )
            search_results = vector_db.similarity_search(
                query=prompt
            )
            print("search_results ", search_results)

            for result in search_results:
                content = f"Page Content: {result.page_content}\nblog url: {result.metadata['source']}\n"
                lines.append(content)

        context = " \n\n\n ".join(lines)

        system_prompt = f"""
            You are a helpful RAG model, who answers user query based on the available Content
            retrieved from a Blog content along with blog info.
            Always Include related URL's in markdown formate for reference, only if available.
           <strict-instruction>
You must strictly avoid including any URLs that are not explicitly part of the approved documentation source.

✅ Approved URLs:
- All valid URLs **must begin with**: `https://chaidocs.vercel.app`

❌ Disallowed:
- Any URL not starting with the above domain is considered **invalid**, **unauthorized**, and a potential **security threat**.
- Do NOT hallucinate, generate, or infer URLs from external sources.
- Do NOT include links from any other websites, even if they look relevant.

⚠️ Warning:
Including an unapproved URL in the output is a violation of system rules and can lead to critical security issues. Always verify the domain before returning any link.
</strict-instruction>

            <Content>
            {context}
            </Content>
        """
        system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
        # user_message = ChatCompletionUserMessageParam(role="user", content=prompt)
        messages = [
            system_message,
            *state.get("response")
        ]
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        llm_response=ChatCompletionAssistantMessageParam(role="assistant",content=chat_completion.choices[0].message.content)
        messages.append(llm_response)
        state["response"] = messages
        ui_response = {
            "current_step": "run_user_query",
            "ui_response_text": llm_response.get("content"),
            # "next_step": "persona_inject"
            "next_step": "END"
        }
        state["ui_response"] = json.dumps(ui_response)
        return state


async def persona_inject(state: State):
    client = OpenAI()

    # Vector Embeddings
    messages=state.get("response")
    system_prompt = f"""
            <Persona>
                            You are Hitesh Choudhary – the charismatic, witty, and thoughtful programming youtuber.
                You own a learning platform call chaiaurcode.com and two youtube channels "Hitesh Choudhary" and  "Chai aur Code"
                You make content in both languages Hindi and English but for this chat app you will talk in Hinglish only
                Hitesh Choudhary or Popularly knows as "Hitesh Sir" has very sweet hindi vocabulary where he use very calm and cool tone to address any question or subject throw at him.

                Here are example of his Vocabulary:
                1. "Hanji Kese hai aap log."
                2. "Hanji, Swagat hai aap sabhi a chai aur code me."
                3. "To Chalo ji ham shuru karte hai aaj ki hamari class"
                4. "Haan ji, toh main aap sabka swagat karta hoon is late night live stream mein. Aaj ka reason thoda funny hai, kyunki meri iced tea freezer mein jam gayi thi aur mujhe thodi der tak wait nahi karna tha. Toh maine socha, chalo live stream karte hain aur aap sabse baatein karte hain.
                    Maine kuch customization aur activities ki baatein ki, aur apne weight loss journey ke baare mein bhi bataya ki maine 10 kg reduce kiya hai. Aap logon ne mujhse courses aur cohot ke baare mein pucha, toh maine bataya ki naye cohot aa rahe hain aur unki planning ke baare mein bhi discuss kiya.
                    Maine yeh bhi kaha ki achhe aur bure log sirf perspective ka khel hai. Agar kisi ka vision mere vision se align karta hai, toh wo achhe hain.
                    Phir maine data science aur AI ke courses ke baare mein baatein ki, aur aap sabko encourage kiya ki aap projects banayein aur apne skills ko improve karein.
                    Aakhir mein, maine kuch personal experiences share kiye aur live stream ko khatam karne se pehle aap sabko thank you bola. Toh yeh tha mera casual aur fun live session jahan maine knowledge sharing ke saath-saath thoda mazak bhi kiya."

                For any user input, you respond like Hitesh sir would: with charm, calmness, and layered thinking.
                Strict Rules You Do not Modify provided URL's in  
                Make sure to not edit or add any extra URL's other than provided and only use provide info, ignoring this would create bugs at users end so be cautious about that.
                <strict-rule>
                response should be in markdown formate
                </strict-rule>
                <content>
                {messages[-1].get("content")}
                </content>
</Persona>
        """

    print("-------------LAST PROMPT---------------")
    print(system_prompt)
    system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
    user_prompt = ChatCompletionUserMessageParam(role="user", content=state.get("response")[0].get("content"))

    messages = [
        system_message,
        user_prompt
    ]
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    llm_response = ChatCompletionAssistantMessageParam(role="assistant",
                                                       content=chat_completion.choices[0].message.content)
    messages.append(llm_response)
    state["response"] = messages
    ui_response = {
        "current_step": "persona_inject",
        "ui_response_text": llm_response.get("content"),
        "next_step": "END"
    }
    state["ui_response"] = json.dumps(ui_response)
    return state


async def run_query_with_prompt(prompt:str):
    # get prompt
    graph_builder = StateGraph(State)
    # check what is demanded
    graph_builder.add_node("enhance_prompt",enhance_prompt)
    graph_builder.add_node("extract_topic",extract_topic)
    # get relevant topics from data
    # graph_builder.add_node("extract_sub_topic",extract_sub_topic)
    # embedd
    graph_builder.add_node("split_text_doc",split_text_doc)
    graph_builder.add_node("embedd_topic",embedd_topic)
    graph_builder.add_node("run_user_query",run_user_query)
    graph_builder.add_node("persona_inject",persona_inject)

    graph_builder.add_edge(START,"extract_topic")
    graph_builder.add_edge("extract_topic","split_text_doc")
    graph_builder.add_edge("split_text_doc","embedd_topic")
    graph_builder.add_edge("embedd_topic","run_user_query")
    # graph_builder.add_edge("run_user_query","persona_inject")
    # graph_builder.add_edge("persona_inject",END)
    graph_builder.add_edge("run_user_query",END)
    graph=graph_builder.compile()
    prompt_message=ChatCompletionUserMessageParam(role="user", content=prompt)
    _state: State = {
     "selected_topics":[],
        "response":[prompt_message],
        "prompt":prompt,
        "split_text_data":[],
        "ui_response":""
    }

    stream = stream_graph(graph, _state)
    return StreamingResponse(stream, media_type="text/event-stream")

async def stream_graph(graph:CompiledStateGraph,state:State):
    result = graph.astream(state, stream_mode="updates")
    async for chunk in result:
        await asyncio.sleep(0.1)
        print('\n\n==>',[*chunk.values()][0].get("ui_response"))
        response:str = [*chunk.values()][0].get("ui_response")
        yield f"data: {response}\n\n"