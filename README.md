# 🤖 AI Due Diligence

A professional, enterprise-grade Streamlit application for automated due diligence document analysis with AI-powered insights, checklist matching, and intelligent Q&A capabilities.

## ✨ Features

### 🎯 **Hierarchical Project Navigation**
- Two-level selection: Project → Data Room
- Smart project discovery from `data/vdrs/` structure
- Document count statistics for each data room
- Support for multiple companies per project

### 📊 **Intelligent Checklist Matching**
- **Enhanced AI Matching**: LLM-generated descriptions for each checklist item explain what documents should satisfy requirements
- **Semantic Understanding**: Uses both original checklist text and AI descriptions for richer document matching
- Automated document-to-checklist mapping with improved accuracy
- PRIMARY/ANCILLARY relevance tagging
- Dynamic relevancy thresholds
- Clean, compact display with download buttons and expandable AI descriptions
- Real-time filtering without reprocessing

### ❓ **Due Diligence Questions**
- Pre-configured question lists from `data/questions/`
- Automated answer extraction from documents
- AI-powered comprehensive answers
- Document relevance scoring
- Source document citations with downloads

### 💬 **Interactive Q&A with Citations**
- Free-form question asking
- 16 pre-configured quick questions across 4 categories:
  - Financial & Performance
  - Legal & Compliance
  - Business & Operations
  - Risk & Strategy
- Precise document citations with excerpts
- AI agent synthesis of answers

### 📈 **Strategic Analysis**
- Company overview generation
- Strategic alignment assessment
- Risk identification from missing documents
- Go/No-Go recommendations
- Export strategic reports

### 🤖 **AI Enhancement (Optional)**
- Powered by Anthropic Claude (Haiku/Sonnet/Opus)
- **Checklist Description Generation**: AI creates detailed explanations for each checklist item
- Document summarization with batch processing
- **Enhanced Semantic Matching**: Combines document summaries with LLM-generated checklist descriptions
- Natural language understanding and synthesis
- Comprehensive error handling and retry logic
- Toggle AI features on/off for comparison

## 🚀 Quick Start

### Prerequisites
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd dd_poc
```

### Running Locally
```bash
# Option 1: Use the run script (recommended)
./run.sh

# Option 2: Manual uv commands
uv sync                           # Install dependencies
uv run streamlit run app.py       # Run the app

# Option 3: Development mode with auto-reload
uv run streamlit run app.py --server.runOnSave true
```

### Environment Setup (for AI features)
```bash
# Create .env file in the project directory
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env

# Optional: Set environment for configuration
echo "ENVIRONMENT=development" >> .env
echo "TOKENIZERS_PARALLELISM=false" >> .env
```

### Verification
```bash
# Test that the app imports correctly
uv run python -c "from app import DDChecklistApp; print('✅ App ready')"

# Start the application to verify everything works
uv run streamlit run app.py
```

## 📱 User Interface

### Sidebar Layout
1. **🎯 Select Project** - Choose from available M&A projects
2. **📁 Select Data Room** - Pick specific company within project
3. **🚀 Process Data Room** - Start analysis
4. **⚙️ Configuration** - AI settings and options

### Main Tabs
1. **📈 Summary & Analysis**
   - Strategy selector with preview
   - Company overview (AI-generated)
   - Strategic alignment analysis
   - Export capabilities

2. **📊 Checklist Matching**
   - Checklist selector with preview
   - **AI-generated descriptions** for each checklist item (when AI enabled)
   - Category progress bars
   - Document relevance indicators
   - Adjustable thresholds
   - Download buttons for each document

3. **❓ Due Diligence Questions**
   - Question list selector
   - Categorized question display
   - Source document listing
   - AI answer generation

4. **💬 Q&A with Citations**
   - Free-form question input
   - Quick question buttons
   - Source excerpts
   - Download links

## 📁 Project Structure

```
dd_poc/
├── app.py                     # 🎯 Main Streamlit application
├── src/                       # 📦 Modular architecture
│   ├── __init__.py           # Package initialization & exports
│   ├── config.py             # Configuration management
│   ├── ai_integration.py     # AI/LangGraph integration
│   ├── document_processing.py # Document operations & processing
│   ├── services.py           # Business logic services
│   ├── ui_components.py      # Reusable UI components
│   └── utils.py              # Error handling & utilities
├── data/                      # 📊 Data directories
│   ├── checklist/           # Due diligence checklists (.md)
│   ├── questions/           # Question lists (.md)
│   ├── strategy/            # Strategic documents (.md)
│   └── vdrs/               # Virtual Data Rooms (5 projects)
│       ├── automated-mobile-robotics-expansion/
│       ├── industrial-ai-dominance/
│       ├── industrial-security-leadership/
│       ├── proj-ra-1/
│       └── technology-led-services-transformation/
├── requirements.txt           # Python dependencies (for reference)
├── pyproject.toml            # uv project configuration
├── run.sh                    # 🚀 Launch script
├── .env                      # API keys (create this)
├── .venv/                    # uv virtual environment (auto-created)
└── .logs/                   # Application logs (auto-created)
```

## 🎨 Key Features Explained

### Document Processing
- **Supported Formats**: PDF, DOCX, DOC, TXT, MD
- **Chunking**: Smart 500-char chunks with 50-char overlap
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: In-memory FAISS for fast retrieval

### Relevance Scoring
- **Primary Documents**: ≥50% relevance (🔹 PRIMARY tag)
- **Ancillary Documents**: <50% relevance (🔸 ANCILLARY tag)
- **Adjustable Thresholds**: Real-time filtering
- **No Document Limits**: Shows all relevant matches

### AI Capabilities
- **Checklist Description Generation**: Creates detailed explanations for what documents satisfy each requirement
- **Document Summarization**: Brief summaries for categorization
- **Enhanced Semantic Matching**: Combines document summaries with checklist descriptions for better accuracy
- **Strategic Analysis**: Alignment with M&A objectives
- **Question Answering**: Comprehensive responses with context
- **Company Overview**: Executive summary generation

### Export Options
- **Strategic Reports**: Markdown format with full analysis
- **Company Summaries**: Structured overview documents
- **Document Downloads**: Direct file access from UI

## 🌐 Deployment

### Streamlit Cloud (Recommended - Free)
1. Fork/push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Add ANTHROPIC_API_KEY in Streamlit secrets
5. Deploy (automatic)

### Local Development
```bash
# Install dependencies (automatically creates virtual environment)
uv sync

# Run with hot reload for development
uv run streamlit run app.py --server.runOnSave true

# Add new dependencies
uv add <package-name>

# Update dependencies
uv lock --upgrade
```

## 💡 Usage Tips

### For Best Results
1. **Organize Documents**: Use logical folder structures
2. **Descriptive Names**: Clear, meaningful file names
3. **Complete Data Rooms**: Include all relevant documents
4. **Specific Checklists**: Detailed, unambiguous items
5. **Enable AI Features**: Use AI descriptions for significantly improved matching accuracy

### Performance Optimization
- First run downloads AI model (~90MB)
- Subsequent runs use cached model
- Processing speed: ~10-20 documents/second
- Use relevancy thresholds to filter results

### Checklist Format
```markdown
## A. Category Name
1. First item to check
2. Second item to check
3. Third item to check

## B. Another Category
1. Another checklist item
2. More items to verify
```

### Question Format
```markdown
## Category Name
- Question one?
- Question two?
- Question three?
```

## 🔧 Configuration

### Sidebar Settings
- **AI Features Toggle**: Enable/disable AI enhancements
- **API Key Input**: For Anthropic Claude access

### Tab-Specific Controls
- **Relevancy Threshold**: Filter document matches (0.2-0.8)
- **Primary Threshold**: Classify as primary/ancillary (0.3-0.9)
- **Preview Expanders**: View selected content

## 📈 Use Cases

- **M&A Due Diligence**: Comprehensive deal evaluation
- **Compliance Audits**: Regulatory document review
- **Risk Assessment**: Gap analysis and identification
- **Contract Analysis**: Agreement review and extraction
- **Investment Evaluation**: Strategic fit assessment

## 🛠️ Troubleshooting

### Debug Tools
```bash
# Test application imports
uv run python -c "from app import DDChecklistApp; app = DDChecklistApp(); print('✅ App working')"

# Check project structure
ls -la src/

# Clean Python cache files
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

### Common Issues
1. **"No projects found"**: Check `data/vdrs/` folder structure
2. **"No checklists found"**: Add `.md` files to `data/checklist/`
3. **"AI packages not available"**: Run `uv sync` to install dependencies
4. **"API key not found"**: Create `.env` file with ANTHROPIC_API_KEY
5. **Import errors**: Clean cache files with the command above
6. **Tokenizer warnings**: Already fixed with `TOKENIZERS_PARALLELISM=false` in `.env`

### Performance
- Large data rooms (>100 docs) may take 2-3 minutes
- Use progress bars to monitor processing
- Check logs in `.logs/` directory for detailed information

## 📝 License

MIT License - See LICENSE file for details

## 🏗️ Architecture

This application uses a modular architecture:

- **`app.py`**: Main Streamlit application
- **`src/`**: All modules organized by responsibility
  - **`config.py`**: Configuration management
  - **`ai_integration.py`**: AI/LangGraph integration for Claude
  - **`document_processing.py`**: File handling and text extraction
  - **`services.py`**: Business logic (parsing, matching, Q&A)
  - **`ui_components.py`**: Reusable Streamlit components
  - **`utils.py`**: Error handling, logging, and utilities

## 🤝 Contributing

Contributions welcome!

## 📧 Support

For questions or support:
1. Check the [troubleshooting section](#-troubleshooting)
2. Test your setup: `uv run python -c "from app import DDChecklistApp; print('✅ Ready')"`
3. Open an issue on GitHub

---

**Built with ❤️ using Streamlit, LangGraph, and Anthropic Claude**