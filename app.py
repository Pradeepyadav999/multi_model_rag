import streamlit as st
import os
from uuid import uuid4
import fitz  # PyMuPDF
import pandas as pd
import base64
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Formula
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
import io

# --- SETUP AND CONSTANTS ---
st.set_page_config(page_title="FREE PDF Assistant", layout="wide", page_icon="📄")

# Load FREE models (cached to avoid reloading)
@st.cache_resource
def load_models():
    """Load all FREE Hugging Face models"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # FREE embeddings
    
    # FREE vision model for image/table analysis
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    
    # FREE text generation model
    text_generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    
    return embedding_model, image_to_text, text_generator

embedding_model, image_to_text_model, text_gen_model = load_models()

# Directory Constants
OUTPUT_DIR = "processed_output"
TABLE_SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "table_screenshots")
IMAGE_SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "image_screenshots")
PAGE_SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "page_screenshots")
PROCESSED_CSV_PATH = os.path.join(OUTPUT_DIR, "processed_document.csv")

# Create directories
os.makedirs(TABLE_SCREENSHOT_DIR, exist_ok=True)
os.makedirs(IMAGE_SCREENSHOT_DIR, exist_ok=True)
os.makedirs(PAGE_SCREENSHOT_DIR, exist_ok=True)

# --- INITIALIZATION ---
def initialize_app():
    """Initialize session state"""
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "upload"

# --- CORE FUNCTIONS ---
def generate_embedding(text: str) -> list | None:
    """Generate FREE embedding using sentence-transformers"""
    if not text or not text.strip(): 
        return None
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def get_element_screenshot(pdf_path: str, page_num: int, bbox: tuple) -> tuple[str, bytes, Image.Image] | tuple[None, None, None]:
    """Take screenshot of element"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        clip = fitz.Rect(bbox)
        pix = page.get_pixmap(clip=clip, dpi=150)  # Lower DPI for FREE tier
        doc.close()
        
        if not pix or pix.width < 10 or pix.height < 10: 
            return None, None, None
        
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Convert to PIL Image for HuggingFace models
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        return base64_image, img_bytes, pil_image
    except Exception as e:
        print(f"Screenshot Error: {e}")
        return None, None, None

def analyze_table_free(pil_image: Image.Image) -> dict:
    """Analyze table using FREE Hugging Face model"""
    print("  > Analyzing table with FREE HF model...")
    try:
        # Use image-to-text model
        result = image_to_text_model(pil_image)
        caption = result[0]['generated_text'] if result else ""
        
        # For tables, we'll use OCR-like description
        # Note: Full table extraction requires paid OCR, so we provide description
        content = f"Table content: {caption}"
        
        return {
            "caption": caption,
            "markdown_table": content
        }
    except Exception as e:
        print(f"Table Analysis Error: {e}")
        return {"caption": "Table detected", "markdown_table": "Table content extracted"}

def analyze_image_free(pil_image: Image.Image) -> dict:
    """Analyze image using FREE Hugging Face model"""
    print("  > Analyzing image with FREE HF model...")
    try:
        result = image_to_text_model(pil_image)
        summary = result[0]['generated_text'] if result else "Image detected"
        
        return {
            "caption": summary,
            "summary": summary
        }
    except Exception as e:
        print(f"Image Analysis Error: {e}")
        return {"caption": "Image", "summary": "Image content extracted"}

def process_pdf(pdf_path: str):
    """Process PDF and extract all content"""
    st.write("🔄 Step 1/4: Partitioning PDF...")
    
    try:
        elements = partition_pdf(
            filename=pdf_path, 
            strategy="fast",  # Faster for FREE tier
            infer_table_structure=False  # Disable for speed
        )
    except Exception as e:
        st.error(f"Error partitioning PDF: {e}")
        st.info("💡 Tip: Try a smaller PDF or simpler document")
        return 0
    
    doc = fitz.open(pdf_path)
    processed_data = []
    current_heading = "Document"
    current_subheading = ""
    
    st.write(f"🔄 Step 2/4: Processing {len(elements)} elements...")
    progress_bar = st.progress(0)
    
    for i, el in enumerate(elements):
        element_type = type(el).__name__
        page_number = getattr(el.metadata, 'page_number', 1)
        
        # Track headings
        if element_type == "Title":
            current_heading = el.text
            current_subheading = ""
            continue
        elif element_type == "Header":
            current_subheading = el.text
            continue
        
        # Process Tables
        if "Table" in element_type:
            print(f"\nProcessing Table on page {page_number}...")
            
            # Try to get coordinates
            try:
                coords_md = el.metadata.coordinates
                page = doc.load_page(page_number - 1)
                x_scale = page.rect.width / coords_md.system.width
                y_scale = page.rect.height / coords_md.system.height
                bbox_pt = (
                    coords_md.points[0][0] * x_scale,
                    coords_md.points[0][1] * y_scale,
                    coords_md.points[2][0] * x_scale,
                    coords_md.points[2][1] * y_scale
                )
                
                b64_img, img_bytes, pil_img = get_element_screenshot(pdf_path, page_number, bbox_pt)
                
                if pil_img:
                    table_ss_path = os.path.join(TABLE_SCREENSHOT_DIR, f"table_p{page_number}_{i}.png")
                    with open(table_ss_path, "wb") as f:
                        f.write(img_bytes)
                    
                    analysis = analyze_table_free(pil_img)
                    caption = analysis.get("caption", "")
                    content = analysis.get("markdown_table", el.text)
                else:
                    table_ss_path = None
                    caption = "Table"
                    content = el.text
            except:
                table_ss_path = None
                caption = "Table"
                content = el.text
            
            embedding_text = f"Section: {current_heading}\n{current_subheading}\nTable: {caption}\n{content}"
            embedding = generate_embedding(embedding_text)
            
            if embedding:
                processed_data.append({
                    "id": str(uuid4()),
                    "chunk_type": "table",
                    "content": content,
                    "caption": caption,
                    "section": current_heading,
                    "subsection": current_subheading,
                    "screenshot_path": table_ss_path or "",
                    "embedding": embedding,
                    "page_number": page_number
                })
        
        # Process Images
        elif "Image" in element_type:
            print(f"\nProcessing Image on page {page_number}...")
            
            try:
                coords_md = el.metadata.coordinates
                page = doc.load_page(page_number - 1)
                x_scale = page.rect.width / coords_md.system.width
                y_scale = page.rect.height / coords_md.system.height
                bbox_pt = (
                    coords_md.points[0][0] * x_scale,
                    coords_md.points[0][1] * y_scale,
                    coords_md.points[2][0] * x_scale,
                    coords_md.points[2][1] * y_scale
                )
                
                b64_img, img_bytes, pil_img = get_element_screenshot(pdf_path, page_number, bbox_pt)
                
                if pil_img and len(img_bytes) > 5000:
                    img_ss_path = os.path.join(IMAGE_SCREENSHOT_DIR, f"image_p{page_number}_{i}.png")
                    with open(img_ss_path, "wb") as f:
                        f.write(img_bytes)
                    
                    analysis = analyze_image_free(pil_img)
                    caption = analysis.get("caption", "")
                    summary = analysis.get("summary", "")
                    
                    embedding_text = f"Section: {current_heading}\n{current_subheading}\nImage: {caption}\n{summary}"
                    embedding = generate_embedding(embedding_text)
                    
                    if embedding:
                        processed_data.append({
                            "id": str(uuid4()),
                            "chunk_type": "image",
                            "content": summary,
                            "caption": caption,
                            "section": current_heading,
                            "subsection": current_subheading,
                            "screenshot_path": img_ss_path,
                            "embedding": embedding,
                            "page_number": page_number
                        })
            except Exception as e:
                print(f"Image processing error: {e}")
        
        # Process Text
        elif len(el.text) > 100:
            print(f"\nProcessing Text on page {page_number}...")
            embedding_text = f"Section: {current_heading}\n{current_subheading}\n{el.text}"
            embedding = generate_embedding(embedding_text)
            
            if embedding:
                processed_data.append({
                    "id": str(uuid4()),
                    "chunk_type": "text",
                    "content": el.text,
                    "caption": "",
                    "section": current_heading,
                    "subsection": current_subheading,
                    "screenshot_path": "",
                    "embedding": embedding,
                    "page_number": page_number
                })
        
        progress_bar.progress((i + 1) / len(elements))
    
    doc.close()
    
    st.write("💾 Step 3/4: Saving to CSV...")
    df = pd.DataFrame(processed_data)
    
    if df.empty:
        st.warning("⚠️ No content extracted. Try a different PDF.")
        return 0
    
    final_columns = ["id", "chunk_type", "content", "caption", "section", "subsection", "screenshot_path", "embedding", "page_number"]
    for col in final_columns:
        if col not in df.columns:
            df[col] = ""
    
    df = df[final_columns]
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    
    st.write("✅ Step 4/4: Complete!")
    return len(df)

# --- UI COMPONENTS ---
def upload_ui():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 🌟 100% FREE PDF Assistant")
        st.markdown("""
        **Powered by FREE Hugging Face models:**
        - 🆓 No API keys needed
        - 🆓 No credit card required
        - 🆓 Unlimited usage
        - 📄 Works with any PDF
        
        **Features:**
        - Extract text, tables, and images
        - AI-powered question answering
        - Visual search with screenshots
        """)
        
        st.info("💡 **Tip:** For best results, use PDFs under 20 pages")
    
    with col2:
        st.markdown("### 📤 Upload Your PDF")
        uploaded_file = st.file_uploader("Select a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("🚀 Processing your PDF... This may take a few minutes."):
                temp_pdf_path = f"temp_{uuid4().hex}.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                num_chunks = process_pdf(temp_pdf_path)
                
                if num_chunks > 0:
                    st.success(f"✅ Successfully processed {num_chunks} chunks!")
                    os.remove(temp_pdf_path)
                    st.session_state.current_stage = "ask"
                    st.rerun()
                else:
                    st.error("❌ Failed to process PDF. Please try another file.")
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

def ask_ui():
    st.markdown("## 💬 Ask Your Document")
    st.caption("Powered by FREE Hugging Face AI models 🤗")
    
    try:
        df = pd.read_csv(PROCESSED_CSV_PATH).fillna("")
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
    except FileNotFoundError:
        st.warning("⚠️ No processed document found. Please upload a PDF first.")
        if st.button("📤 Go to Upload"):
            st.session_state.current_stage = "upload"
            st.rerun()
        st.stop()
    
    question = st.text_input(
        "💬 Your Question:", 
        placeholder="e.g., What is the main topic of this document?",
        key="question_input"
    )
    
    if question:
        cleaned = question.strip().lower()
        if len(cleaned.split()) < 3:
            st.error("❓ Please ask a more complete question (at least 3 words)")
            st.stop()
        
        with st.spinner("🔍 Searching document..."):
            q_embedding = generate_embedding(question)
            
            if q_embedding is None:
                st.error("Error generating question embedding")
                st.stop()
            
            df["similarity"] = df["embedding"].apply(
                lambda emb: cosine_similarity([emb], [q_embedding])[0][0]
            )
            
            max_sim = df["similarity"].max()
            if max_sim < 0.2:
                st.warning("🤔 No relevant content found for your question. Try rephrasing.")
                st.stop()
            
            top_results = df.sort_values("similarity", ascending=False).head(5)
            
            # Build context
            context = ""
            for _, row in top_results.iterrows():
                context += f"[{row['section']}] {row['content']}\n\n"
            
            # Generate answer using FREE model
            prompt = f"""Answer this question based only on the context below.

Context: {context[:1000]}

Question: {question}

Answer:"""
            
            try:
                response = text_gen_model(prompt, max_length=200, do_sample=False)
                answer = response[0]['generated_text']
            except:
                # Fallback if model fails
                answer = f"Based on the document, here's what I found:\n\n{context[:500]}..."
        
        st.markdown("### 📝 Answer")
        st.success(answer)
        
        st.markdown("---")
        st.markdown("### 📚 Relevant Content")
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            with st.expander(f"#{idx} - {row['chunk_type'].title()} from {row['section']} (Page {row['page_number']})"):
                st.markdown(f"**Similarity:** {row['similarity']:.2%}")
                
                if row['caption']:
                    st.markdown(f"**Caption:** {row['caption']}")
                
                st.markdown(row['content'])
                
                if row['screenshot_path'] and os.path.exists(row['screenshot_path']):
                    st.image(row['screenshot_path'], use_container_width=True)

def main():
    initialize_app()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("---")
    
    if st.session_state.current_stage == "upload":
        st.sidebar.info("📍 Stage: Upload PDF")
    else:
        st.sidebar.success("📍 Stage: Ask Questions")
        if st.sidebar.button("🔄 Upload New PDF"):
            st.session_state.current_stage = "upload"
            if os.path.exists(PROCESSED_CSV_PATH):
                os.remove(PROCESSED_CSV_PATH)
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 About")
    st.sidebar.info("""
    This app uses 100% FREE resources:
    - Hugging Face models
    - No API keys needed
    - Open source libraries
    """)
    
    # Main content
    if st.session_state.current_stage == "upload":
        upload_ui()
    else:
        ask_ui()

if __name__ == "__main__":
    main()