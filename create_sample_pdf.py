"""
Create a sample PDF document for testing the RAG application
"""

import os

# Try to import reportlab, but handle gracefully if not available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("⚠️  reportlab not available. Will create text file instead.")
    REPORTLAB_AVAILABLE = False

def create_sample_pdf(filename="sample_document.pdf"):
    """
    Create a sample PDF document about AI and Machine Learning
    """
    if not REPORTLAB_AVAILABLE:
        print("❌ reportlab not available. Cannot create PDF.")
        return None
    
    print(f"📝 Creating sample PDF: {filename}")
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    # Content
    content = []
    
    # Title
    title = Paragraph("Introduction to Artificial Intelligence and Machine Learning", title_style)
    content.append(title)
    content.append(Spacer(1, 20))
    
    # Sections
    sections = [
        {
            "title": "What is Artificial Intelligence?",
            "text": """
            Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
            that are programmed to think and learn like humans. The term may also be applied to any 
            machine that exhibits traits associated with a human mind such as learning and 
            problem-solving. AI systems can perform tasks that typically require human intelligence, 
            such as visual perception, speech recognition, decision-making, and translation between 
            languages.
            """
        },
        {
            "title": "Machine Learning Fundamentals",
            "text": """
            Machine Learning (ML) is a subset of AI that provides systems the ability to automatically 
            learn and improve from experience without being explicitly programmed. ML focuses on the 
            development of computer programs that can access data and use it to learn for themselves. 
            The process of learning begins with observations or data, such as examples, direct experience, 
            or instruction, to look for patterns in data and make better decisions in the future.
            """
        },
        {
            "title": "Types of Machine Learning",
            "text": """
            There are three main types of machine learning:
            
            1. Supervised Learning: Uses labeled training data to learn a function that maps inputs to outputs.
            
            2. Unsupervised Learning: Finds hidden patterns or intrinsic structures in input data without labeled examples.
            
            3. Reinforcement Learning: An agent learns to behave in an environment by performing actions and receiving rewards.
            
            Each type has its specific use cases and applications in various domains.
            """
        },
        {
            "title": "Applications of AI and ML",
            "text": """
            AI and ML have numerous applications across different industries:
            
            - Healthcare: Medical diagnosis, drug discovery, personalized treatment
            - Finance: Fraud detection, algorithmic trading, risk assessment
            - Transportation: Autonomous vehicles, route optimization, traffic management
            - Entertainment: Recommendation systems, content generation, game AI
            - Retail: Customer service chatbots, inventory management, price optimization
            - Manufacturing: Quality control, predictive maintenance, supply chain optimization
            """
        },
        {
            "title": "Natural Language Processing",
            "text": """
            Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
            interpret, and manipulate human language. NLP draws from many disciplines, including 
            computer science and computational linguistics, to help computers understand human 
            language in a valuable way. Modern NLP techniques include sentiment analysis, 
            machine translation, question answering, and text summarization.
            """
        },
        {
            "title": "Deep Learning",
            "text": """
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in data. 
            These neural networks attempt to simulate the behavior of the human brain to "learn" 
            from large amounts of data. Deep learning has been particularly successful in areas 
            such as image recognition, speech recognition, and natural language processing.
            """
        },
        {
            "title": "Future of AI",
            "text": """
            The future of AI holds tremendous potential for transforming various aspects of human life. 
            We can expect to see advancements in areas such as general AI, quantum computing integration, 
            more sophisticated robotics, and ethical AI development. However, with these advancements 
            come challenges including job displacement, privacy concerns, and the need for responsible 
            AI governance. The key is to develop AI systems that are beneficial, safe, and aligned 
            with human values.
            """
        }
    ]
    
    # Add sections to document
    for section in sections:
        # Section title
        section_title = Paragraph(section["title"], styles['Heading2'])
        content.append(section_title)
        content.append(Spacer(1, 12))
        
        # Section text
        section_text = Paragraph(section["text"], styles['Normal'])
        content.append(section_text)
        content.append(Spacer(1, 20))
    
    # Footer
    footer_text = """
    This document was automatically generated for testing purposes in a Retrieval-Augmented 
    Generation (RAG) system. It contains information about artificial intelligence and machine 
    learning concepts that can be used for question-answering demonstrations.
    """
    footer = Paragraph(footer_text, styles['Normal'])
    content.append(Spacer(1, 30))
    content.append(footer)
    
    # Build PDF
    doc.build(content)
    print(f"✅ Sample PDF created successfully: {filename}")
    print(f"📊 File size: {os.path.getsize(filename)} bytes")
    
    return filename

def create_sample_pdf_simple(filename="sample_document.txt"):
    """
    Create a simple text file if reportlab is not available
    """
    print(f"📝 Creating sample text file: {filename}")
    
    content = """
Introduction to Artificial Intelligence and Machine Learning

What is Artificial Intelligence?

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.

Machine Learning Fundamentals

Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future.

Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning: Uses labeled training data to learn a function that maps inputs to outputs.

2. Unsupervised Learning: Finds hidden patterns or intrinsic structures in input data without labeled examples.

3. Reinforcement Learning: An agent learns to behave in an environment by performing actions and receiving rewards.

Each type has its specific use cases and applications in various domains.

Applications of AI and ML

AI and ML have numerous applications across different industries:

- Healthcare: Medical diagnosis, drug discovery, personalized treatment
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, route optimization, traffic management
- Entertainment: Recommendation systems, content generation, game AI
- Retail: Customer service chatbots, inventory management, price optimization
- Manufacturing: Quality control, predictive maintenance, supply chain optimization

Natural Language Processing

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, to help computers understand human language in a valuable way. Modern NLP techniques include sentiment analysis, machine translation, question answering, and text summarization.

Deep Learning

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. These neural networks attempt to simulate the behavior of the human brain to "learn" from large amounts of data. Deep learning has been particularly successful in areas such as image recognition, speech recognition, and natural language processing.

Future of AI

The future of AI holds tremendous potential for transforming various aspects of human life. We can expect to see advancements in areas such as general AI, quantum computing integration, more sophisticated robotics, and ethical AI development. However, with these advancements come challenges including job displacement, privacy concerns, and the need for responsible AI governance. The key is to develop AI systems that are beneficial, safe, and aligned with human values.

This document was automatically generated for testing purposes in a Retrieval-Augmented Generation (RAG) system. It contains information about artificial intelligence and machine learning concepts that can be used for question-answering demonstrations.
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Sample text file created successfully: {filename}")
    print(f"📊 File size: {os.path.getsize(filename)} bytes")
    
    return filename

def main():
    """Main function to create sample documents"""
    print("📄 Sample Document Creator for RAG Testing")
    print("=" * 50)
    
    try:
        # Try to create PDF first
        if REPORTLAB_AVAILABLE:
            filename = create_sample_pdf("sample_ai_document.pdf")
            print(f"💡 Use this file path in the RAG application: {os.path.abspath(filename)}")
        else:
            raise ImportError("reportlab not available")
    except (ImportError, Exception):
        print("❌ reportlab not installed. Creating text file instead...")
        print("💡 To create PDF: py -m pip install reportlab")
        filename = create_sample_pdf_simple("sample_ai_document.txt")
        print(f"💡 Use this file path in the RAG application: {os.path.abspath(filename)}")
        print("🔧 Note: You'll need to modify the loader for text files")

if __name__ == "__main__":
    main()
