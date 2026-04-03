from multiprocessing import process
import re
import os
from pathlib import Path
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from dotenv import load_dotenv
load_dotenv()


# 1. PATHS
BASE_DIR= Path(os.getenv("DATA_DIR"))

MASKED_DIR= BASE_DIR/"masked"
EXTRACTED_DIR = Path("data/extracted_md")


for path in [MASKED_DIR]:
    path.mkdir(parents=True,exist_ok=True)


class SecureShield:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.patterns = {
            "SEC_ID": r"\d{10}-\d{2}-\d{6}", # Commission File Number
            "ZIP_CODE": r"\b\d{5}(?:-\d{4})?\b"
        }
        
    def sanitize_text(self,text):
        operators= {
            "PERSON": OperatorConfig("replace",{"new_value":"[ENITITY_REDACTED]"}),
            "PHONE_NUMBER":OperatorConfig("mask",{"chars_to_mask":10,"masking_char":"*","from_end":True})
        }
        results=self.analyzer.analyze(text=text,language='en',entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS","LOCATION"])
        
        #redact it
        anonymized_res = self.anonymizer.anonymize(text=text, analyzer_results=results,operators=operators)
        return anonymized_res.text

    def mask_text(self,text):
        masked_text=text
        for label,pattern in self.patterns.items():
            masked_text= re.sub(pattern,f"[{label}_REDACTED]",masked_text)

        return masked_text
    
    def mask_single_file(self, md_path: Path):
        """Processes ONLY the specific file passed to it"""
        if not md_path.exists():
            print(f"Error: {md_path} not found!")
            return None

        # Find all .md files recursively
        print(f"Shielding: {md_path.name}")
        
        # 1. Read the specific file
        raw_content = md_path.read_text(encoding='utf-8')
      
      
            
        #apply multi layer protection
        safe_content =self.sanitize_text(raw_content)
        #rule asaed
        safe_content= self.mask_text(safe_content)
        
        #save to masked directory
        subfolder= md_path.parent.name
        target_dir= MASKED_DIR/subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        output_filename= md_path.name.replace(".md","_masked.md")
        output_path = target_dir / output_filename
        output_path.write_text(safe_content, encoding="utf-8")

        return {
            "doc_name":md_path,
            "safe_content":safe_content,
            "output_path": Path(output_path)
         
        }

if __name__ == "__main__":
    shield = SecureShield()
    
    # LOCAL BATCH MODE:
    # Look for all extracted files and process them one by one
    extracted_files = list(EXTRACTED_DIR.rglob("*.md"))
    
    for file_path in extracted_files:
        shield.mask_single_file(file_path)