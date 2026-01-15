import fitz  # PyMuPDF
import re

class PDFService:
    def parse_pdf(self, file_path):
        """
        Parses PDF into chunks with chapter information and Markdown tables.
        """
        doc = fitz.open(file_path)
        clauses = []
        current_chapter = "未分类"
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Simple heuristic for chapters: lines that look like "第一章", "1.", "1.1"
            lines = text.split('\n')
            page_content = []
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Update current chapter if line matches a pattern
                if re.match(r'^第[一二三四五六七八九十百]+章', line) or re.match(r'^\d+\.', line):
                    current_chapter = line
                
                page_content.append(line)
            
            # Identify tables - PyMuPDF has find_tables (since v1.23.0)
            tables = page.find_tables()
            table_markdowns = []
            for table in tables:
                df = table.to_pandas()
                md = df.to_markdown(index=False)
                table_markdowns.append(md)
            
            # Combine text and tables
            full_content = "\n".join(page_content)
            for md in table_markdowns:
                full_content += f"\n\n{md}\n"
            
            # In a real scenario, we'd split full_content by clause
            # Here we treat each page or large block as a clause for simplicity
            clauses.append({
                "chapter_path": current_chapter,
                "content": full_content
            })
            
        return clauses

pdf_service = PDFService()
