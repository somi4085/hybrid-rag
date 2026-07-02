from typing import List, Dict


class DocumentLoader:

    def load_documents(self, file_path:str):

        with open(file_path, "r") as f:
            lines = f.readlines()

        return [{
            "id":1,
            "text" : line.strip()} for i, line in enumerate(lines)]