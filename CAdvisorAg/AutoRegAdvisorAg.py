pip install streamlit requests PyPDF2 beautifulsoup4 pandas groq langgraph langchain-groq
import os
import re
import requests
import io
import PyPDF2
import streamlit as st
import pandas as pd
from groq import Groq
from bs4 import BeautifulSoup
from langchain.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Groq client with the provided API key
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Updated automotive regulatory websites
REGULATORY_WEBSITES = {
    "US": "https://www.nhtsa.gov/laws-regulations/fmvss",
    "EU": "https://unece.org/transport/vehicle-regulations",
    "China": "https://www.cccauthorization.com/ccc-certification/automotive-regulations",
    "India": "https://bis.gov.in/index.php/standards/technical-department/transport-engineering/",
    "Australia": "https://www.infrastructure.gov.au/infrastructure-transport-vehicles/vehicles/vehicle-design-regulation/australian-design-rules"
}

# Define node states
class AgentState(BaseModel):
    query: str = Field(default="")
    market: str = Field(default="")
    selected_url: str = Field(default="")
    pdf_urls: list = Field(default_factory=list)
    pdf_contents: dict = Field(default_factory=dict)
    final_answer: str = Field(default="")

# Define state graph nodes
def get_market(state):
    """Determine which market to look for regulatory documents."""
    logger.info("Starting market determination...")
    query = state.query
    
    prompt = f"""
    Based on the following query, determine which automotive regulatory market the user is interested in (US, EU, China, India, or Australia).
    If the market is not clear, respond with "UNCLEAR".
    
    Query: {query}
    
    Return only the market name or "UNCLEAR" without any additional text.
    """
    
    try:
        # Call LLM to determine market
        logger.info("Calling LLM to determine market...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        
        market = response.choices[0].message.content.strip()
        logger.info(f"Market determined: {market}")
        
        if market == "UNCLEAR":
            return {"market": "UNCLEAR"}
        else:
            return {"market": market}
    except Exception as e:
        logger.error(f"Error determining market: {str(e)}")
        return {"market": "UNCLEAR"}

def select_url(state):
    """Select the appropriate regulatory website based on market."""
    logger.info("Selecting URL based on market...")
    market = state.market
    
    if market in REGULATORY_WEBSITES:
        selected_url = REGULATORY_WEBSITES[market]
        logger.info(f"Selected URL: {selected_url}")
        return {"selected_url": selected_url}
    else:
        logger.warning(f"Market {market} not found in regulatory websites")
        return {"selected_url": "UNCLEAR"}

def extract_pdf_links(state):
    """Extract PDF links from the regulatory website."""
    logger.info("Extracting PDF links...")
    url = state.selected_url
    query = state.query
    
    try:
        logger.info(f"Fetching content from {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links on the page
        links = soup.find_all('a')
        
        # Extract PDF links
        pdf_links = []
        for link in links:
            href = link.get('href')
            if href and href.endswith('.pdf'):
                full_url = href if href.startswith('http') else (url + href if not url.endswith('/') else url + '/' + href)
                if link.text:
                    pdf_links.append((link.text.strip(), full_url))
        
        logger.info(f"Found {len(pdf_links)} PDF links")
        
        if not pdf_links:
            logger.warning("No PDF links found, using default sample links")
            # If no PDFs found, use some sample links for demonstration
            pdf_links = [
                ("Automotive Safety Regulation", "https://example.com/auto_safety.pdf"),
                ("Emission Standards", "https://example.com/emissions.pdf"),
                ("Vehicle Type Approval", "https://example.com/type_approval.pdf")
            ]
        
        # Use LLM to select relevant PDFs based on the query
        prompt = f"""
        Based on the user query: "{query}", select the most relevant PDF documents from the following list.
        Return the indices of the selected documents (0-based) as a comma-separated list.
        
        PDFs:
        {pd.DataFrame(pdf_links, columns=['Title', 'URL']).to_string()}
        
        Return only the indices as a comma-separated list, without any additional text.
        """
        
        logger.info("Calling LLM to select relevant PDFs...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Extract indices from response
        indices_str = response.choices[0].message.content.strip()
        logger.info(f"LLM response for PDF selection: {indices_str}")
        
        indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
        
        # Get selected PDFs
        selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
        logger.info(f"Selected {len(selected_pdfs)} PDFs")
        
        return {"pdf_urls": selected_pdfs}
    
    except Exception as e:
        logger.error(f"Error extracting PDF links: {str(e)}")
        return {"pdf_urls": []}

def download_and_process_pdfs(state):
    """Download PDFs and extract content."""
    logger.info("Downloading and processing PDFs...")
    pdf_urls = state.pdf_urls
    pdf_contents = {}
    
    for title, url in pdf_urls:
        try:
            logger.info(f"Downloading PDF: {title} from {url}")
            # For demonstration, we'll simulate PDF content if the URL is not accessible
            try:
                response = requests.get(url, timeout=10)
                pdf_file = io.BytesIO(response.content)
                
                # Read PDF content
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except:
                logger.warning(f"Could not download PDF from {url}, using simulated content")
                # Simulate PDF content for demonstration
                text = f"This is simulated content for {title}. The actual PDF could not be downloaded or processed."
                text += "\n\nThis document covers automotive regulations including safety standards, emission requirements, "
                text += "and compliance procedures for vehicle manufacturers."
            
            pdf_contents[title] = text
            logger.info(f"Successfully processed PDF: {title}")
        
        except Exception as e:
            logger.error(f"Error processing PDF {title}: {str(e)}")
            pdf_contents[title] = f"Error processing PDF: {str(e)}"
    
    return {"pdf_contents": pdf_contents}

def analyze_content(state):
    """Analyze PDF content and generate answer."""
    logger.info("Analyzing content...")
    query = state.query
    pdf_contents = state.pdf_contents
    
    # Combine all contents
    combined_text = ""
    for title, content in pdf_contents.items():
        combined_text += f"--- Document: {title} ---\n{content}\n\n"
    
    # Split text into chunks to stay within token limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,  # Adjust based on token limit
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(combined_text)
    logger.info(f"Split content into {len(chunks)} chunks")
    
    # Process chunks and collect insights
    insights = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        prompt = f"""
        I'm analyzing automotive regulatory documents to answer a user's query.
        
        User query: {query}
        
        Document text (chunk {i+1}/{len(chunks)}):
        {chunk}
        
        Extract key insights relevant to the query from this text chunk.
        """
        
        try:
            logger.info(f"Calling LLM for chunk {i+1} analysis...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            insights.append(response.choices[0].message.content)
            logger.info(f"Successfully analyzed chunk {i+1}")
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            insights.append(f"Error processing chunk {i+1}: {str(e)}")
    
    # Combine insights and generate final answer
    combined_insights = "\n\n".join(insights)
    
    prompt = f"""
    Based on the following insights extracted from automotive regulatory documents, provide a comprehensive answer to the user's query.
    
    User query: {query}
    
    Insights from documents:
    {combined_insights}
    
    Provide a well-structured, accurate, and cohesive answer focusing on automotive regulations.
    """
    
    try:
        logger.info("Generating final answer...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        final_answer = response.choices[0].message.content
        logger.info("Final answer generated successfully")
    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}")
        final_answer = f"Error generating final answer: {str(e)}"
    
    return {"final_answer": final_answer}

# Define the state graph
def build_graph():
    logger.info("Building state graph...")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_market", get_market)
    workflow.add_node("select_url", select_url)
    workflow.add_node("extract_pdf_links", extract_pdf_links)
    workflow.add_node("download_and_process_pdfs", download_and_process_pdfs)
    workflow.add_node("analyze_content", analyze_content)
    
    # Add edges
    workflow.add_edge("get_market", "select_url")
    workflow.add_conditional_edges(
        "get_market",
        lambda state: "select_url" if state.market != "UNCLEAR" else "get_market",
        {
            "select_url": "select_url",
            "get_market": "get_market"
        }
    )
    workflow.add_edge("select_url", "extract_pdf_links")
    workflow.add_edge("extract_pdf_links", "download_and_process_pdfs")
    workflow.add_edge("download_and_process_pdfs", "analyze_content")
    workflow.add_edge("analyze_content", END)
    
    logger.info("State graph built successfully")
    return workflow.compile()

# Streamlit UI
def main():
    st.set_page_config(page_title="Automotive Regulatory Document Assistant", layout="wide")
    st.title("Automotive Regulatory Document Assistant")
    
    # Sidebar for logs
    st.sidebar.title("Execution Logs")
    log_placeholder = st.sidebar.empty()
    
    # Create a log handler that writes to the streamlit sidebar
    log_output = []
    
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_record = self.format(record)
            log_output.append(log_record)
            log_placeholder.text('\n'.join(log_output))
    
    # Add the streamlit handler to the logger
    streamlit_handler = StreamlitLogHandler()
    logger.addHandler(streamlit_handler)
    
    # API key is already set, so we don't need to ask for it
    st.success("API key is pre-configured. Ready to use!")
    
    # Initialize graph
    logger.info("Initializing application...")
    try:
        graph = build_graph()
        logger.info("Graph initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing graph: {str(e)}")
        st.error(f"Error initializing graph: {str(e)}")
        return
    
    # User query
    query = st.text_input("Enter your automotive regulatory query:")
    
    # Market selection
    market_options = list(REGULATORY_WEBSITES.keys())
    selected_market = st.selectbox("Select a market (or let the system detect it):", ["Auto-detect"] + market_options)
    
    if st.button("Process Query"):
        if query:
            with st.spinner("Processing your query..."):
                logger.info(f"Processing query: {query}")
                
                # Set market if manually selected
                market = "" if selected_market == "Auto-detect" else selected_market
                
                # Run the graph
                try:
                    logger.info(f"Starting graph execution with market: {market or 'Auto-detect'}")
                    state = AgentState(query=query, market=market)
                    result = graph.invoke(state)
                    logger.info("Graph execution completed")
                    
                    # Display results
                    st.subheader("Results")
                    
                    if result.market != "UNCLEAR":
                        st.write(f"Market: {result.market}")
                    else:
                        st.error("Could not determine market automatically. Please select a market.")
                        market_options = list(REGULATORY_WEBSITES.keys())
                        selected_market = st.selectbox("Please select a market:", market_options)
                        if st.button("Confirm Market"):
                            state = AgentState(query=query, market=selected_market)
                            result = graph.invoke(state)
                    
                    st.write(f"Website: {result.selected_url}")
                    
                    st.subheader("Documents Analyzed")
                    if result.pdf_urls:
                        for title, url in result.pdf_urls:
                            st.write(f"- {title} ([link]({url}))")
                    else:
                        st.write("No documents were found or selected.")
                    
                    st.subheader("Answer")
                    st.write(result.final_answer)
                except Exception as e:
                    logger.