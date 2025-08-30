# 🤖 AI Due Diligence

A powerful Streamlit application for automated due diligence document analysis with AI-powered insights, checklist matching, and intelligent Q&A capabilities.

## ✨ Features

### 🎯 **Hierarchical Project Navigation**
- Two-level selection: Project → Data Room
- Smart project discovery from `data/vdrs/` structure
- Document count statistics for each data room
- Support for multiple companies per project

### 📊 **Intelligent Checklist Matching**
- Automated document-to-checklist mapping
- PRIMARY/ANCILLARY relevance tagging
- Dynamic relevancy thresholds
- Clean, compact display with download buttons
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
- Powered by Anthropic Claude (Haiku/Sonnet)
- Document summarization
- Intelligent matching
- Natural language understanding
- Toggle AI features on/off for comparison

## 🚀 Quick Start

### Prerequisites
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd dm_agents/dd-checklist
```

### Running Locally
```bash
# Option 1: From parent directory
cd /Users/jmzlx/Projects/dm_agents
uv run --project dd-checklist streamlit run dd-checklist/app.py

# Option 2: From dd-checklist directory
cd dd-checklist
uv run streamlit run app.py

# Option 3: Use the run script
./run.sh
```

### Environment Setup (for AI features)
```bash
# Create .env file in dd-checklist directory
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
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
dd-checklist/
├── app.py                 # Main Streamlit application
├── langgraph_config.py    # LangGraph agent configuration
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── run.sh               # Launch script
├── .env                 # API keys (create this)
├── data/
│   ├── checklist/       # Due diligence checklists (.md)
│   │   ├── bloomberg.md
│   │   └── expanded.md
│   ├── questions/       # Question lists (.md)
│   │   └── questions.md
│   ├── strategy/        # Strategic documents (.md)
│   │   └── projects.md
│   └── vdrs/           # Virtual Data Rooms
│       ├── automated-mobile-robotics-expansion/
│       │   ├── navifloor-robotics-inc/
│       │   └── polar-dynamics-robotics-inc/
│       ├── industrial-ai-dominance/
│       │   └── nexus-industrial-intelligence-inc/
│       └── ... (other projects)
└── .streamlit/
    └── config.toml      # UI configuration
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
- **Document Summarization**: Brief summaries for categorization
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
# Install dependencies
uv pip install -r requirements.txt

# Run with hot reload
uv run streamlit run app.py --server.runOnSave true
```

## 💡 Usage Tips

### For Best Results
1. **Organize Documents**: Use logical folder structures
2. **Descriptive Names**: Clear, meaningful file names
3. **Complete Data Rooms**: Include all relevant documents
4. **Specific Checklists**: Detailed, unambiguous items

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

### Common Issues
1. **"No projects found"**: Check `data/vdrs/` folder structure
2. **"No checklists found"**: Add `.md` files to `data/checklist/`
3. **"AI packages not available"**: Install with `uv pip install -r requirements.txt`
4. **"API key not found"**: Create `.env` file with ANTHROPIC_API_KEY

### Performance
- Large data rooms (>100 docs) may take 2-3 minutes
- Use progress bars to monitor processing
- Check console for detailed logs

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## 📧 Support

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit, LangGraph, and Anthropic Claude**