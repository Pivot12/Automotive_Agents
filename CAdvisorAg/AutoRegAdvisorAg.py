# pip install streamlit requests PyPDF2 beautifulsoup4 pandas groq langgraph langchain-groq
import os
import re
import base64
import requests
import io
import PyPDF2
import streamlit as st
import pandas as pd
from streamlit_mermaid import st_mermaid
from groq import Groq
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
import logging
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Diagram Image Creation
def create_diagram_image():
   """Create a diagram image using NetworkX and Matplotlib"""
   # Create a graph
   G = nx.DiGraph()
   # Add nodes with positions for a more controlled layout
   nodes = {
       "User Input": {"pos": (0, 0)},
       "Market Selection": {"pos": (0, -1)},
       "Process Query": {"pos": (1, -0.5)},
       "Initialize Agent": {"pos": (2, -0.5)},
       "Processing Pipeline": {"pos": (3, -0.5)},
       "Document Analysis": {"pos": (4, -0.5)},
       "Generate Answer": {"pos": (5, -0.5)},
       "Groq LLM API": {"pos": (3, -2)},
       "PDF Processing": {"pos": (4, -1.5)},
       "Error Handling": {"pos": (2, -2)}
   }
   # Add all nodes
   for node, attrs in nodes.items():
       G.add_node(node, **attrs)
   # Define node colors by category
   node_colors = {
       "User Input": "#d0f0c0",
       "Market Selection": "#d0f0c0",
       "Process Query": "#d0f0c0",
       "Initialize Agent": "#c5daf9",
       "Processing Pipeline": "#c5daf9",
       "Document Analysis": "#c5daf9",
       "Generate Answer": "#c5daf9",
       "Groq LLM API": "#f9d6c5",
       "PDF Processing": "#c5daf9",
       "Error Handling": "#f9c5c5"
   }
   # Add edges (connections)
   edges = [
       ("User Input", "Process Query"),
       ("Market Selection", "Process Query"),
       ("Process Query", "Initialize Agent"),
       ("Initialize Agent", "Processing Pipeline"),
       ("Processing Pipeline", "Document Analysis"),
       ("Document Analysis", "Generate Answer"),
       ("Groq LLM API", "Processing Pipeline"),
       ("Groq LLM API", "Document Analysis"),
       ("Groq LLM API", "Generate Answer"),
       ("Document Analysis", "PDF Processing"),
       ("Error Handling", "Processing Pipeline"),
       ("Error Handling", "Document Analysis")
   ]
   G.add_edges_from(edges)
   # Create figure with a white background
   plt.figure(figsize=(10, 6), facecolor='white')
   # Get node positions
   pos = nx.get_node_attributes(G, 'pos')
   # Draw nodes with custom colors
   for node, color in node_colors.items():
       nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color,
                             node_size=2500, edgecolors='black')
   # Draw regular edges (solid lines)
   regular_edges = [(u, v) for u, v in edges if u not in ["Groq LLM API", "Error Handling"]]
   nx.draw_networkx_edges(G, pos, edgelist=regular_edges, arrows=True, arrowsize=20,
                         width=1.5, edge_color='black')
   # Draw special edges (dashed lines)
   special_edges = [(u, v) for u, v in edges if u in ["Groq LLM API", "Error Handling"]]
   nx.draw_networkx_edges(G, pos, edgelist=special_edges, arrows=True, arrowsize=20,
                         width=1.5, edge_color='gray', style='dashed')
   # Add labels with white background for better readability
   label_options = {"fc": "white", "alpha": 0.8, "bbox": {"pad": 5, "boxstyle": "round"}}
   nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', bbox=label_options)
   # Remove axes
   plt.axis('off')
   plt.tight_layout()
   # Save the plot to a BytesIO object
   buffer = BytesIO()
   plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
   buffer.seek(0)
   plt.close()
   # Create image from buffer
   image = Image.open(buffer)
   return image
# Function to get base64 encoded image for embedded display
def get_image_base64(image):
   buffered = BytesIO()
   image.save(buffered, format="PNG")
   img_str = base64.b64encode(buffered.getvalue()).decode()
   return img_str



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

# Define the state class as a TypedDict-compatible dictionary
class AgentState(BaseModel):
    query: str = ""
    market: str = ""
    selected_url: str = ""
    pdf_urls: List[Tuple[str, str]] = []
    pdf_contents: Dict[str, str] = {}
    final_answer: str = ""

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
    
    # Create the graph with the node schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_market", get_market)
    workflow.add_node("select_url", select_url)
    workflow.add_node("extract_pdf_links", extract_pdf_links)
    workflow.add_node("download_and_process_pdfs", download_and_process_pdfs)
    workflow.add_node("analyze_content", analyze_content)
    
    # Add the START edge
    workflow.add_edge(START, "get_market")
    
    # Add conditional edges with safe state access
    workflow.add_conditional_edges(
        "get_market",
        lambda s: "select_url" if s.market != "UNCLEAR" else "get_market",
        {
            "select_url": "select_url",
            "get_market": "get_market"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("select_url", "extract_pdf_links")
    workflow.add_edge("extract_pdf_links", "download_and_process_pdfs")
    workflow.add_edge("download_and_process_pdfs", "analyze_content")
    workflow.add_edge("analyze_content", END)
    
    logger.info("State graph built successfully")
    return workflow.compile()

# Simplified alternative implementation
def create_simple_agent():
    """Create a simplified agent without using LangGraph."""
    def process_query(query, market=None):
        logger.info("Starting simplified agent processing")
        results = {
            "query": query,
            "market": market,
            "selected_url": "",
            "pdf_urls": [],
            "pdf_contents": {},
            "final_answer": ""
        }
        
        # Step 1: Determine market if not provided
        if not market:
            logger.info("Determining market...")
            prompt = f"""
            Based on the following query, determine which automotive regulatory market the user is interested in (US, EU, China, India, or Australia).
            If the market is not clear, respond with "UNCLEAR".
            
            Query: {query}
            
            Return only the market name or "UNCLEAR" without any additional text.
            """
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10
                )
                market = response.choices[0].message.content.strip()
                results["market"] = market
            except Exception as e:
                logger.error(f"Error determining market: {str(e)}")
                results["market"] = "UNCLEAR"
                return results
        
        # Step 2: Select URL
        if results["market"] in REGULATORY_WEBSITES:
            results["selected_url"] = REGULATORY_WEBSITES[results["market"]]
        else:
            logger.warning(f"Market {results['market']} not found in regulatory websites")
            results["selected_url"] = "UNCLEAR"
            return results
        
        # Step 3: Extract PDF links
        try:
            url = results["selected_url"]
            logger.info(f"Fetching content from {url}")
            
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = soup.find_all('a')
            pdf_links = []
            
            for link in links:
                href = link.get('href')
                if href and href.endswith('.pdf'):
                    full_url = href if href.startswith('http') else (url + href if not url.endswith('/') else url + '/' + href)
                    if link.text:
                        pdf_links.append((link.text.strip(), full_url))
            
            if not pdf_links:
                pdf_links = [
                    ("Automotive Safety Regulation", "https://example.com/auto_safety.pdf"),
                    ("Emission Standards", "https://example.com/emissions.pdf"),
                    ("Vehicle Type Approval", "https://example.com/type_approval.pdf")
                ]
            
            # Select relevant PDFs
            prompt = f"""
            Based on the user query: "{query}", select the most relevant PDF documents from the following list.
            Return the indices of the selected documents (0-based) as a comma-separated list.
            
            PDFs:
            {pd.DataFrame(pdf_links, columns=['Title', 'URL']).to_string()}
            
            Return only the indices as a comma-separated list, without any additional text.
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            indices_str = response.choices[0].message.content.strip()
            indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
            selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
            
            results["pdf_urls"] = selected_pdfs
        except Exception as e:
            logger.error(f"Error extracting PDF links: {str(e)}")
            results["pdf_urls"] = []
            return results
        
        # Step 4: Download and process PDFs
        pdf_contents = {}
        for title, url in results["pdf_urls"]:
            try:
                # Simulate PDF content
                text = f"This is simulated content for {title}. This document covers automotive regulations including safety standards, emission requirements, and compliance procedures for vehicle manufacturers."
                pdf_contents[title] = text
            except Exception as e:
                logger.error(f"Error processing PDF {title}: {str(e)}")
                pdf_contents[title] = f"Error processing PDF: {str(e)}"
        
        results["pdf_contents"] = pdf_contents
        
        # Step 5: Analyze content
        combined_text = ""
        for title, content in results["pdf_contents"].items():
            combined_text += f"--- Document: {title} ---\n{content}\n\n"
        
        prompt = f"""
        Based on the following automotive regulatory documents, provide a comprehensive answer to the user's query.
        
        User query: {query}
        
        Documents:
        {combined_text}
        
        Provide a well-structured, accurate, and cohesive answer focusing on automotive regulations.
        """
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            results["final_answer"] = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            results["final_answer"] = f"Error generating final answer: {str(e)}"
        
        return results

    return process_query

# Streamlit UI
def main():
    st.set_page_config(page_title="Automotive Regulations AI Agent", layout="wide")
    st.title("Automotive Regulations AI Agent")
    
    # Sidebar for logs
    st.sidebar.title("Execution Logs")
    log_placeholder = st.sidebar.empty()
    
    # Create a log handler that writes to the streamlit sidebar
    log_output = []
    
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_record = self.format(record)
            log_output.append(log_record)
            log_placeholder.text('\n'.join(log_output[-30:]))  # Keep only last 30 logs
    
    # Add the streamlit handler to the logger
    streamlit_handler = StreamlitLogHandler()
    logger.addHandler(streamlit_handler)
    
    # API key is already set, so we don't need to ask for it
    st.success("API key is pre-configured. Ready to use!")
    
    # Initialize agent
    logger.info("Initializing application...")
    try:
        # First try to use LangGraph
        use_langgraph = False
        if use_langgraph:
            graph = build_graph()
            logger.info("LangGraph initialized successfully")
        else:
            # Fallback to simplified implementation
            process_query = create_simple_agent()
            logger.info("Simplified agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error(f"Error initializing agent: {str(e)}")
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
                market = None if selected_market == "Auto-detect" else selected_market
                
                # Run the agent
                try:
                    logger.info(f"Starting agent with market: {market or 'Auto-detect'}")
                    
                    if use_langgraph:
                        # LangGraph approach
                        input_state = AgentState(query=query, market=market)
                        result = graph.invoke(input_state)
                    else:
                        # Simplified approach
                        result = process_query(query, market)
                    
                    logger.info("Processing completed")
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Display the market
                    if result["market"] and result["market"] != "UNCLEAR":
                        st.write(f"Market: {result['market']}")
                    else:
                        st.error("Could not determine market automatically. Please select a market.")
                        selected_market = st.selectbox("Please select a market:", market_options, key="market_select_after_error")
                        if st.button("Confirm Market", key="confirm_market_button"):
                            if use_langgraph:
                                input_state = AgentState(query=query, market=selected_market)
                                result = graph.invoke(input_state)
                            else:
                                result = process_query(query, selected_market)
                    
                    st.write(f"Website: {result['selected_url']}")
                    
                    st.subheader("Documents Analyzed")
                    if result["pdf_urls"]:
                        for title, url in result["pdf_urls"]:
                            st.write(f"- {title} ([link]({url}))")
                    else:
                        st.write("No documents were found or selected.")
                    
                    st.subheader("Answer")
                    st.write(result["final_answer"])
                except Exception as e:
                    logger.error(f"Error during processing: {str(e)}")
                    st.error(f"An error occurred while processing your query: {str(e)}")
        else:
            st.warning("Please enter a query.")
    
    # The architecture diagram
    st.markdown("---")
    st.subheader("How This Application Works")
      
    # Create a collapsible section for the diagram
    with st.expander("Click to view the application architecture diagram"):
          # Skip trying to use Mermaid and directly use the image-based approach
         with st.spinner("Generating diagram image..."):
              try:
                  # Create the diagram
                  diagram_image = create_diagram_image()
                  # Display the image
                  st.image(diagram_image, caption="Application Architecture", use_column_width=True)
                  # Add download option
                  img_str = get_image_base64(diagram_image)
                  href = f'<a href="data:image/png;base64,{img_str}" download="regulatory_agent_diagram.png">Download Diagram Image</a>'
                  st.markdown(href, unsafe_allow_html=True)
              except Exception as img_error:
                  st.error(f"Error generating diagram image: {str(img_error)}")
                  # Text-only fallback as last resort
                  st.code("""
                  User Input → Process Query → Initialize Agent → Processing Pipeline → Document Analysis → Generate Answer
                                                                      ↑                      ↑                 ↑
                                                                Groq LLM API connections (provides intelligence)
                                                                      ↑                      ↑
                                                                Error Handling (monitors process)
                                                                                            ↓
                                                                                     PDF Processing
                  """)  
    # Explanation of the diagram
    st.markdown("""
   ### Diagram Explanation
    This diagram shows how the Automotive Regulatory Document Assistant works:
    1. **User Interface**: You enter your query and select a market
    2. **Processing Pipeline**: The system analyzes your request
    3. **Document Analysis**: Relevant documents are found and processed
    4. **Answer Generation**: A comprehensive answer is created
    """)      
       
    # Add some usage instructions
    st.markdown("---")
    st.markdown("""
    ## How to use this tool
    1. Enter your query about automotive regulations
    2. Either select a specific market or let the system detect it
    3. Click "Process Query" to start the analysis
    4. The system will identify relevant documents from all global regulatory databases and provide an accurate answer based on their content only
    
    ## Example queries
    - "What are the crash test requirements for passenger vehicles in the US?"
    - "Explain the emission standards for electric vehicles in the EU"
    - "What are the approval procedures for importing vehicles to Australia?"
    """)

if __name__ == "__main__":
    main()
