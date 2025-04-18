# # Import Libraries
import numpy as np
import tempfile
import soundfile as sf
import gradio as gr
import json
import random
import os
from dotenv import load_dotenv
import logging

from google.api_core.exceptions import GoogleAPIError, NotFound, PermissionDenied
from google import genai
from google.genai import types

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()  # Load variables from .env.example file
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env.example file.")

# ------------- Global State -------------
full_transcript = []
lecture_index = []
vector_db = None
rag_chain = None
current_correct_answer = ""

LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

CSS_PATH = "css/style.css"


# ------------- Transcript State Management -------------
def get_full_transcript_text():
    return " ".join(full_transcript)


def clear_transcript_data():
    global full_transcript
    full_transcript = []
    print("Transcript data cleared.")
    return ""


# ------------- Gemini STT Function -------------
def transcribe_audio_chunk(audio_path):
    try:
        client = genai.Client(api_key=API_KEY)
        uploaded_file = client.files.upload(file=audio_path)
        print(f"✅ File uploaded. URI: {uploaded_file.uri}, Name: {uploaded_file.name}")

        prompt = (
            "Please perform speech-to-text transcription for the provided audio file. "
            "Output the transcribed text followed by the key points as a numbered list. "
            "Do not use any JSON formatting—just return plain text."
        )

        print("🚀 Sending transcription request to Gemini...")
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[prompt, uploaded_file]
        )

        if not response.candidates:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            return f"Transcription failed. Block Reason: {block_reason}"

        candidate = response.candidates[0]
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            transcript = candidate.content.parts[0].text
            print("✅ Transcription successful.")
            return transcript
        else:
            return "Error: Failed to parse transcription response."

    except PermissionDenied as e:
        return f"❌ Permission Denied: {e.message}"
    except NotFound as e:
        return f"❌ Resource Not Found: {e.message}"
    except GoogleAPIError as e:
        return f"❌ API Error: {e.message}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"


# ------------- Formatting the Output -------------
def format_transcription_result(result_text):
    return result_text


# ------------- Gradio Transcription Handler -------------
def handle_transcription_request(audio_file):
    if audio_file is None:
        return "", get_full_transcript_text(), gr.update(
            value=None), "Transcription not initiated.", "Input declined. No audio file provided."

    transcript_text = transcribe_audio_chunk(audio_file)
    formatted_chunk = format_transcription_result(transcript_text)
    full_transcript.append(formatted_chunk)

    return (
        formatted_chunk,
        get_full_transcript_text(),
        gr.update(value=None),
        "Transcription successful.",
        "Input accepted. Audio file is being processed."
    )


def handle_clear_transcript():
    clear_transcript_data()
    return "", "Transcript cleared."


# ------------- Chunking, Embedding, Vector DB & RAG -------------

def chunk_transcript(text, chunk_size: int = 800, overlap_size: int = 150):
    # Optionally, you could call: text = correct_transcript_errors(text)
    document = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size
    )
    chunks = splitter.split_documents(documents=document)
    print(f"File split into {len(chunks)} chunks.")
    return chunks


def create_vector_db(text_chunks, collection_name="transcription-rag"):
    global vector_db
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
        vector_db = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            collection_name=collection_name
        )
        print(f"Vector DB created with collection_name: {collection_name}")
        return vector_db
    except Exception as e:
        raise Exception(f"Error creating vector DB: {str(e)}")


def setup_rag_chain(vector_db):
    if not vector_db:
        raise ValueError("Vector DB not initialized!")

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=API_KEY
        )

        # Few-shot query rewriting prompt
        query_prompt = PromptTemplate.from_template("""
            You are an AI assistant that helps rephrase queries.

            Example 1:
            Original Question: Who is Master Sito?
            Alternative Queries:
              1. According to the transcript, what is Master Sito's role?
              2. What does the transcript state about Master Sito?
              3. How is Master Sito described in the lecture?

            Example 2:
            Original Question: Who is Master Sito?
            Even if the transcript contains a minor typo (e.g., 'Master Ceto'),
            assume the intended name is Master Sito.

            Now, given the original question: {question}
            Generate three alternative queries:
        """)

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
            llm=llm,
            prompt=query_prompt
        )

        # Main prompt for answering with grounding and few-shot examples
        main_template = """
            You are an educational assistant. Answer the user's question based solely on the transcript context provided.
            Disregard minor transcription errors (for example, if the transcript has "Master Ceto" but context indicates it should be "Master Sito").
            If the answer is explicitly stated, provide it exactly. Otherwise, reply "I don’t know."

            Few-shot examples:
            ---------------------
            Transcript Example 1:
            "Master Sito said: 'Face life with humor.'"
            Q: What did Master Sito say about life?
            A: Face life with humor.
            ---------------------
            Transcript Example 2:
            "According to the lecture, Master Sito is a monk living in seclusion."
            Q: Who is Master Sito?
            A: He is a monk.
            ---------------------
            Now, using the transcript below:
            Transcript:
            {context}

            Question: {question}
            Answer:
        """

        prompt = ChatPromptTemplate.from_template(template=main_template)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        print("RAG chain setup complete!")
        return chain

    except Exception as e:
        raise Exception(f"Error setting up RAG chain: {str(e)}")


def handle_indexing_request(transcript_text):
    global vector_db, rag_chain
    if not transcript_text or len(transcript_text.strip()) == 0:
        return "⚠️ Transcript is empty. Please transcribe or paste something first."
    try:
        chunks = chunk_transcript(transcript_text)
        vector_db = create_vector_db(chunks)
        rag_chain = setup_rag_chain(vector_db)
        return f"✅ Indexing complete. {len(chunks)} chunks indexed."
    except Exception as e:
        return f"❌ Indexing failed: {str(e)}"


def query(chain, question: str):
    if not chain:
        print("RAG chain not initialized!")
    try:
        return chain.invoke(question)
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")


def answer_query_using_rag(user_query):
    global rag_chain
    if not rag_chain:
        return "⚠️ Please index the transcript first."
    try:
        result = query(rag_chain, user_query)
        return f"💬 {result}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# -------------Quiz Generation with few shot prompting---------------------------------------------
def setup_quiz_chain():
    try:
        llm_quiz = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            # Consider setting a reasonable max_tokens limit, e.g., max_tokens=1024
            max_tokens=None,
            # Consider setting an explicit timeout, e.g., timeout=120
            timeout=None,
            max_retries=2,
            google_api_key=API_KEY
        )

        quiz_template = """
            You are an educational assistant. Your task is to generate 5 multiple-choice quiz questions based only on the transcript provided below.
            Please return the output strictly as valid JSON. Do not include any introductory text or markdown formatting around the JSON object.
            The JSON should be a list containing 5 objects, each following this format:

            {{
              "question": "Your quiz question here.",
              "options": ["Option A", "Option B", "Option C", "Option D"],
              "answer": "The correct option (must exactly match one of the options)"
            }}

            Transcript:
            {transcript}

            JSON Output:
        """  # Added "JSON Output:" hint and refined instructions slightly

        quiz_prompt = PromptTemplate.from_template(quiz_template)
        # For standard JSON:
        parser = JsonOutputParser()

        # Update the chain to use the JsonOutputParser
        chain = (
                {"transcript": RunnablePassthrough()}
                | quiz_prompt
                | llm_quiz
                | parser  # <-- Use JsonOutputParser instead of StrOutputParser
        )
        print("Quiz chain setup complete!")
        return chain

    except Exception as e:
        raise Exception(f"Error setting up Quiz chain: {str(e)}")


# --- Global Quiz State ---
quiz_state = None


# --- Function to Generate Quiz ---
def generate_quiz(transcript: str):
    global quiz_state
    if not transcript or transcript.strip() == "":
        return "⚠️ Please provide a transcript.", [], "No quiz generated."
    try:
        chain = setup_quiz_chain()
        output = chain.invoke({"transcript": transcript})
        print("DEBUG - Chain output:", output)
        quiz_data = output  # Already parsed JSON from JsonOutputParser.
    except Exception as e:
        return f"Quiz generation failed: {str(e)}", [], "Error occurred."
    if not quiz_data or len(quiz_data) == 0:
        return "⚠️ No quiz questions returned by the model.", [], ""

    # Initialize quiz state with an additional 'answered' flag.
    quiz_state = {
        "questions": quiz_data,
        "current_index": 0,
        "score": 0,
        "streak": 0,  # New: Track consecutive correct answers.
        "answered": False  # New: Flag to indicate if the current question is answered.
    }

    first_question = quiz_data[0]
    return first_question["question"], first_question["options"], ""


# --- Function to Evaluate Answer (without advancing to next question) ---
def select_answer(index: int):
    global quiz_state
    if not quiz_state or "questions" not in quiz_state:
        return "No quiz generated. Please generate a quiz first.", "N/A", "N/A", "N/A", "N/A", "⚠️", "Score: 0 | Streak: 0"

    # Prevent re-answering if the question was already answered.
    if quiz_state.get("answered", False):
        current_question = quiz_state["questions"][quiz_state["current_index"]]
        options = current_question.get("options", [])
        btn_labels = [options[i] if i < len(options) else "N/A" for i in range(4)]
        return (current_question["question"], btn_labels[0], btn_labels[1], btn_labels[2], btn_labels[3],
                "You have already answered. Click 'Next Question' to continue.",
                f"Score: {quiz_state.get('score', 0)} | Streak: {quiz_state.get('streak', 0)}")

    current_question = quiz_state["questions"][quiz_state["current_index"]]
    options = current_question.get("options", [])
    if index >= len(options):
        return "Invalid option selected.", "N/A", "N/A", "N/A", "N/A", "Error", f"Score: {quiz_state.get('score', 0)} | Streak: {quiz_state.get('streak', 0)}"

    selected_option = options[index]

    # Check answer and update score and streak.
    if selected_option == current_question["answer"]:
        feedback = "Correct!"
        quiz_state["score"] += 1
        quiz_state["streak"] += 1
    else:
        feedback = f"Incorrect. The correct answer was: {current_question['answer']}."
        quiz_state["streak"] = 0

    quiz_state["answered"] = True  # Mark the question as answered.
    btn_labels = [options[i] if i < len(options) else "N/A" for i in range(4)]
    score_text = f"Score: {quiz_state['score']} | Streak: {quiz_state['streak']}"
    return (current_question["question"], btn_labels[0], btn_labels[1], btn_labels[2], btn_labels[3],
            feedback, score_text)


# --- Function to Advance to the Next Question ---
def advance_to_next_question():
    global quiz_state
    if not quiz_state or "questions" not in quiz_state:
        return "No quiz generated. Please generate a quiz first.", "N/A", "N/A", "N/A", "N/A", "⚠️", "Score: 0 | Streak: 0"

    if not quiz_state.get("answered", False):
        return "Please select an answer before proceeding.", "N/A", "N/A", "N/A", "N/A", "⚠️", f"Score: {quiz_state['score']} | Streak: {quiz_state['streak']}"

    quiz_state["current_index"] += 1
    quiz_state["answered"] = False  # Reset the answered flag.
    if quiz_state["current_index"] < len(quiz_state["questions"]):
        next_q = quiz_state["questions"][quiz_state["current_index"]]
        options = next_q.get("options", [])
        btn_labels = [options[i] if i < len(options) else "N/A" for i in range(4)]
        return (next_q["question"], btn_labels[0], btn_labels[1], btn_labels[2], btn_labels[3],
                "", f"Score: {quiz_state['score']} | Streak: {quiz_state['streak']}")
    else:
        score = quiz_state["score"]
        total = len(quiz_state["questions"])
        percentage = round((score / total) * 100)
        color = "red" if percentage < 60 else "green"
        # Display final score with some HTML styling.
        percent_display = f"<span style='color:{color}; font-weight:bold;'>{percentage}%</span>"
        final_msg = f"Quiz complete! Your final score is {score} out of {total}: {percent_display}."
        quiz_state = None
        return final_msg, "", "", "", "", "", ""


# --- Combined function to update quiz question & button labels on generation ---
def generate_quiz_and_buttons(transcript: str):
    question, options, feedback = generate_quiz(transcript)
    btn_labels = ["N/A", "N/A", "N/A", "N/A"]
    if isinstance(options, list):
        for i in range(min(len(options), 4)):
            btn_labels[i] = options[i]
    score_text = "Score: 0 | Streak: 0"
    return question, btn_labels[0], btn_labels[1], btn_labels[2], btn_labels[3], feedback, score_text


def select_answer_and_update(index: int):
    # (Call our select_answer function.)
    return select_answer(index)


def load_transcript(full_text):
    # For now, simply return the same text.
    # Adjust this function based on your intended behavior.
    return full_text


def clear_transcript():
    # This dummy implementation clears the transcript and returns a cleared status message.
    return "", "Transcript cleared."


def handle_query_request(user_query):
    if not user_query or not user_query.strip():
        return "⚠️ Please enter a valid question about the lecture."

    # Hypothetical function that uses your indexed transcript + LLM:
    return answer_query_using_rag(user_query)


# ------------------ Gradio Interface with Custom Retro Theme ------------------
with gr.Blocks(
        theme="d8ahazard/material_design_rd",
        css=CSS_PATH  # Load CSS from the external file
) as app:
    # Link Minecraft font and Press Start 2P for retro elements
    # These <link> tags are still useful if the @import in CSS fails or for clarity
    gr.Markdown('<link href="https://fonts.cdnfonts.com/css/minecraft-4" rel="stylesheet">')
    gr.Markdown('<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">')

    # -- Header with Minecraft only on the H2 --
    with gr.Row():
        #  with gr.Column(scale=0, min_width=90):
        #      gr.Image(value="/content/bot.jpg", show_label=False, elem_id="bot-logo", height=90) # Assuming bot.jpg is accessible
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <h2 class="minecraft-heading typewriter neon-text" style="margin: 0;">
                    Inclusive Classroom Assistant
                </h2>
                <p class="neon-text" style="margin: 4px 0 0 0; font-size: 14px;">
                    Upload audio, transcribe, index, and ask anything about your lecture!
                </p>
                """,
                elem_id="header"
            )

    # ------------------ Tab 1: Transcription & Indexing ------------------
    with gr.Tab("🎙️ Transcription & Indexing") as tab1:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<h3 class='neon-text'>Transcription Input</h3>")
                # --- MODIFIED: Removed label text using show_label=False ---
                audio_input = gr.Audio(type="filepath", show_label=False)
                # --- END MODIFICATION ---
                transcribe_button = gr.Button("Transcribe Chunk", elem_classes="accent-bg")
                transcription_input_status_textbox = gr.Textbox(label="Transcription Input Status", lines=1,
                                                                interactive=False)
                latest_chunk_textbox = gr.Textbox(label="Latest Transcript Chunk", lines=10, interactive=False)
                status_textbox = gr.Textbox(label="Status", lines=1, interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("<h3 class='neon-text'>Full Transcript & Indexing</h3>")
                full_transcript_textbox = gr.Textbox(label="Full Lecture Transcript", lines=20, interactive=False)
                with gr.Row():
                    index_button = gr.Button("Index Transcript for Search", elem_classes="accent-bg")
                    clear_button = gr.Button("Clear Full Transcript", elem_classes="accent-bg")
                indexing_status_display = gr.Textbox(label="Indexing Status", lines=2, interactive=False)

    # ------------------ Tab 2: Query Lecture Content ------------------
    with gr.Tab("💬 Query Lecture Content") as tab2:
        gr.Markdown("<h3 class='neon-text'>Ask a question about the lecture content</h3>")
        with gr.Row():
            query_input_textbox = gr.Textbox(
                label="Ask a question",
                placeholder="E.g., What lesson did Sam learn?",
                lines=2
            )
            ask_button = gr.Button("Ask Question", elem_classes="accent-bg")
        # Answer display with neon and retro effects
        answer_display = gr.Markdown(
            "💡 Answer will appear here...",
            elem_classes="query-answer-box retro-panel neon-text",
            # label="Answer" # Markdown doesn't have a label param like this
        )

    # ------------------ Tab 3: Quiz Generator ------------------
    with gr.Tab("📝 Quiz Generator") as tab3:
        # Scoreboard only in this tab with retro neon style
        scoreboard = gr.Markdown("Score: 0 | Streak: 0", elem_id="quiz-scoreboard")
        gr.Markdown("<h3 class='neon-text'>Generate Quiz from Transcript</h3>")
        gr.Markdown(
            "<p class='retro-panel neon-text'>Click <strong>Generate Quiz</strong> to start. Answer each question and review your score and correct answer streak after each question.</p>")
        generate_btn = gr.Button("Generate Quiz", elem_classes="accent-bg")
        quiz_question = gr.Markdown("Question will appear here", elem_classes="retro-panel neon-text")
        with gr.Row():
            option_button1 = gr.Button("Option 1", elem_classes="accent-bg")
            option_button2 = gr.Button("Option 2", elem_classes="accent-bg")
            option_button3 = gr.Button("Option 3", elem_classes="accent-bg")
            option_button4 = gr.Button("Option 4", elem_classes="accent-bg")
        feedback_box = gr.Textbox(label="Feedback", interactive=False, elem_classes="retro-panel neon-text")
        next_btn = gr.Button("Next Question", elem_classes="accent-bg")

    # ------------------ Button Callback Bindings (Placeholder - Add your actual functions) ------------------

    transcribe_button.click(
        fn=handle_transcription_request,
        inputs=[audio_input],
        outputs=[latest_chunk_textbox, full_transcript_textbox, audio_input, status_textbox,
                 transcription_input_status_textbox]
    )
    index_button.click(
        fn=handle_indexing_request,
        inputs=[full_transcript_textbox],
        outputs=[indexing_status_display]
    )
    clear_button.click(
        fn=clear_transcript_data,
        inputs=None,
        outputs=[full_transcript_textbox, status_textbox]
    )
    ask_button.click(
        fn=handle_query_request,
        inputs=[query_input_textbox],
        outputs=[answer_display]
    )
    generate_btn.click(
        fn=generate_quiz_and_buttons,
        inputs=[full_transcript_textbox],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )
    option_button1.click(
        fn=lambda: select_answer_and_update(0),
        inputs=[],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )
    option_button2.click(
        fn=lambda: select_answer_and_update(1),
        inputs=[],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )
    option_button3.click(
        fn=lambda: select_answer_and_update(2),
        inputs=[],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )
    option_button4.click(
        fn=lambda: select_answer_and_update(3),
        inputs=[],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )
    next_btn.click(
        fn=advance_to_next_question,
        inputs=[],
        outputs=[quiz_question, option_button1, option_button2, option_button3, option_button4, feedback_box,
                 scoreboard]
    )

app.launch()
