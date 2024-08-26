from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
import os
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        load_dotenv()

        # Load and split documents
        self.loader = TextLoader('Data/Permissionless_and_Permissioned_Technology-Focused_and_Business_Needs-Driven_Understanding_the_Hybrid_Opportunity_in_Blockchain_Through_a_Case_Study_of_Insolar.txt')
        self.documents = self.loader.load()
        print(f"Total document length: {len(self.documents)} characters")
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)
        print(f"Document split into {len(self.docs)} chunks.")

        # Initialize embeddings and Pinecone client
        self.embeddings = HuggingFaceEmbeddings()
        api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_client = PineconeClient(api_key=api_key)
        self.index_name = "langchain-demo"

        # Setup index
        self.setup_index()

        # Set up the LLM and prompt template
        self.setup_llm()
        self.setup_prompt_template()

    def setup_index(self):
        index_list = self.pinecone_client.list_indexes()
        if self.index_name not in index_list:
            try:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=768,  # Ensure this matches the dimensionality of your embeddings
                    metric='cosine'  # Use the metric that suits your model
                )
                print("Index created successfully.")
            except Exception as e:
                print(f"Failed to create index: {e}")
        else:
            print("Index already exists.")
        self.index = self.pinecone_client.Index(self.index_name)

    def setup_llm(self):
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.8,
            top_k=50
        )

    def setup_prompt_template(self):
        template = """
        You are a knowledgeable expert on bitcoin. These Humans will ask you questions about the article stored. Use the following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Your answer should be short and concise, no longer than two sentences.

        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def get_response(self, question):
        print(f"Received question: {question}")
        query_vector = self.embeddings.embed_documents([question])[0]
        query_result = self.index.query(vector=query_vector, top_k=5)
        print("Full Query result:", query_result)
        matches = query_result.get('matches', [])
        context = " ".join([match.get('values', '') for match in matches if match.get('values')])
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm(formatted_prompt)
        print(f"Response from LLM: {response}")
        return response

# CLI script to interact with the bot
if __name__ == "__main__":
    bot = ChatBot()
    print("Welcome to the World of Bitcoin!")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"Bot: {response}")
