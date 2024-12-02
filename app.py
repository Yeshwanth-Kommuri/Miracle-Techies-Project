import os
import openai
import people_also_ask as paa
from datetime import datetime
from fastapi import FastAPI, HTTPException, Path, Body, Request
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from passlib.context import CryptContext
from typing import List, Dict, Any, Optional, Union
from bson import ObjectId
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from perplexity import call_perplexity
import requests
from bs4 import BeautifulSoup
import validators
import uuid
from fastapi.staticfiles import StaticFiles

# MongoDB connection
client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost/seo_writing_tool')
db = client.seo_writing_tool  # Database name

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load environment variables for API keys
os.environ["OPENAI_API_KEY"] = "sk-RbBLWpktD1GKmaEQK-PdqSxp0Q8ApXEwGt9PteEZptT3BlbkFJXMCKeaiekQZ-bqGKlQBllb3Bl1dG5BCvZVuzjvg3IA"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the application's image directory
def get_image_dir():
    # Get the directory where app.py is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # print(app_dir, "\n-------------------app dir")
    # Create images directory path
    image_dir = os.path.join(app_dir, "images")
    # Create the directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    return image_dir

# Create FastAPI app
app = FastAPI()

# Mount the images directory
image_dir = get_image_dir()
# app.mount("/images", StaticFiles(directory=image_dir), name="images")

# CORS settings for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend development URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)


class ArticleInput(BaseModel):
    language: str
    article_size: str
    tone_of_voice: str
    point_of_view: str
    target_country: str
    target_state: str
    target_city_zip: str
    model: str
    custom_tone_file: Optional[str] = None  # Optional and default to None
    toc: bool
    h3: bool
    quotes: bool
    key_takeaways: bool
    conclusion: bool
    generate_image: bool
    selectedRows: List[Dict[str, Any]]
    user: Dict
    outbound_links: bool = True  # New field for outbound links setting


# Define the input model for the update payload
class ArticleUpdate(BaseModel):
    article: str


# Define the input data model for related questions
class RelatedQuestionInput(BaseModel):
    main_keyword: str
    number_of_questions: int


# Pydantic models
class RegisterUser(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginUser(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr


class ArticleOut(BaseModel):
    id: str
    user: Union[EmailStr, str]
    title: Optional[str] = None
    created_at: Optional[str] = None
    article: Optional[str] = None
    image: Optional[str] = None
    last_modified: Optional[str] = None
    key_words: Optional[str] = None
    is_active: Optional[bool] = True

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class GetArticlesInput(BaseModel):
    email: EmailStr


# Generate article using a different approach to template handling
def create_article_template(data):
    template = (
        "Write a {article_size} article in {language} about {title} and include {keywords} keywords while writing. and also you can take references from input source given by user {inbound_data}. "
        "Tailor the content for readers in {target_country}, {target_state}, {target_city_zip}."
    )

    if data.get("tone_of_voice") == "custom":
        template += "Detect the tone of voice from the user inputed text and write article based on the user tone {custom_tone_file} and from a {point_of_view} perspective."
    else:
        template += "The article should be in a {tone_of_voice} tone and from a {point_of_view} perspective. "

    # Conditional additions
    options = []
    if data.get("toc"):
        options.append("a table of contents")
    if data.get("h3"):
        options.append("H3 headings")
    if data.get("quotes"):
        options.append("relevant quotes")
    if data.get("key_takeaways"):
        options.append("key takeaways")
    if data.get("conclusion"):
        options.append("a summary")

    if options:
        template += " Include " + ", ".join(options) + "."

    content_filters = data.get("content_filters", {})
    # Build prompt with content filter instructions
    content_filter_instructions = []
    
    if content_filters.get("prevent_harmful_content"):
        content_filter_instructions.append(
            "Ensure the content is safe, appropriate, and free from harmful, offensive, or inappropriate material."
        )
    
    if content_filters.get("prevent_competitor_mentions"):
        content_filter_instructions.append(
            "Avoid mentioning or promoting competitor brands and products."
        )
    
    if content_filters.get("ensure_factual_accuracy"):
        content_filter_instructions.append(
            "Verify all statements and claims for factual accuracy. Include only well-researched and verified information."
        )
    
    if content_filters.get("maintain_neutrality"):
        content_filter_instructions.append(
            "Maintain a neutral, unbiased perspective throughout the content."
        )
    
    if content_filters.get("ensure_source_credibility"):
        content_filter_instructions.append(
            "Use and reference only credible, authoritative sources."
        )

    # Add content filter instructions to the main prompt
    content_filter_prompt = "\n".join([
        "Content Filter Requirements:",
        *[f"- {instruction}" for instruction in content_filter_instructions]
    ])

    # Final instruction to exclude introductory text
    template += " Your output should be only the article, without any introductory text."

    # Format the template with provided data
    return template.format(**data)


def get_entities_template(data):
    template = (
        "Can you extract the important key words from the given input text {text} or for the given title {title} of the article"
        "Limit yourself for maximum 5 key words."
        "Your output should be only a list of key words don't include any thing like this Certainly! Here's a short article on."
    )
    return template.format(**data)


# Generate the article based on the input
def generate_entities(answer):
    data = {"title": answer["question"]}
    data["text"] = ""
    if answer["has_answer"]:
        data["text"] = answer["raw_text"]
    template = get_entities_template(data)

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": template}
        ]
        )
    # Extract the generated article text
    entities = response.choices[0].message['content'].strip()
    return entities


# Generate an image using OpenAI DALL-E model
def generate_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    
    # Download the image
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(get_image_dir(), filename)
        
        # Save the image locally
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        
        # Return the local path that will be accessible through the static route
        return f"{image_path}"
    return ""


# Generate the article based on the input
def generate_article(data):
    for each in data["selectedRows"]:
        data["title"] = each["title"]
        data["keywords"] = each["keywords"]
        data["inbound_data"] = extract_text_from_url(each["inboundLink"])
        template = create_article_template(data)

        # Use the selected model
        if data["model"] == "gpt":
            if data["outbound_links"]:
                template += " Also add references that you have considered while writing the article. with 'Article generated with the following citations: ' as a prefix."
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": template}
                ]
            )
            article_text = response.choices[0].message['content'].strip()
            outbound_links = ""

        else:  # llama model
            response = call_perplexity(template)
            article_text = response['choices'][0]['message']['content']
            if data["outbound_links"]:
                outbound_links = "Article generated with the following citations: \n" + "\n".join(response['citations'])
            else:
                outbound_links = ""

        if outbound_links:
            article_text += "\n" + outbound_links

        if each["inboundLink"]:
            article_text += "\nArticle generated from the following Inbound link: \n" + each["inboundLink"]


        # Get the current timestamp
        current_time = datetime.utcnow()

        image_url = ""
        if bool(data["generate_image"]):
            image_url = generate_image(data["title"])

        # Define the document to insert
        document = {
            "user": data["user"]["email"],
            "title": data["title"],
            "created_at": current_time,
            "article": article_text,
            "last_modified": current_time,
            "image": image_url,
            "is_active": True
        }

        # Insert the document into MongoDB
        db.articles.insert_one(document)
    return ""


def extract_text_from_url(url: str) -> Optional[str]:
    """
    Validates URL and extracts text content from the webpage.
    Returns None if URL is invalid or content cannot be extracted.
    """
    try:
        # Validate URL
        if not validators.url(url):
            return None

        # Fetch webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Limit text length to avoid overwhelming the API
        return text[:5000] if text else None

    except Exception as e:
        print(f"Error extracting text from URL: {str(e)}")
        return None


# Registration endpoint
@app.post("/register", response_model=UserOut)
async def register(user: RegisterUser):
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(user.password)
    user_dict = user.dict()
    user_dict['password'] = hashed_password
    result = await db.users.insert_one(user_dict)
    user_out = UserOut(
        id=str(result.inserted_id),
        name=user.name,
        email=user.email
    )
    return user_out


# Login endpoint
@app.post("/login", response_model=UserOut)
async def login(user: LoginUser):
    existing_user = await db.users.find_one({"email": user.email})
    if not existing_user or not pwd_context.verify(user.password, existing_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    user_out = UserOut(
        id=str(existing_user['_id']),
        name=existing_user.get('name'),
        email=existing_user['email']
    )
    return user_out


# POST endpoint for article generation
@app.post("/generate-article")
async def create_article_endpoint(request: Request):
    try:
        data = await request.json()
        
        # Generate the article
        _ = generate_article(data)
        
        return JSONResponse(content={"message": "Article generated successfully"})
        
    except Exception as e:
        print(f"Error generating article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Fetch related questions
def fetch_related_questions(keyword, limit):
    questions = paa.get_related_questions(keyword, limit-1)
    # print(paa.get_answer(questions[0]))
    return [{"question": q, "key_words": generate_entities(paa.get_answer(q)), "answer": paa.get_answer(q)} for q in questions]


# POST endpoint for related questions
@app.post("/get-related-questions")
async def related_questions_endpoint(input_data: RelatedQuestionInput):
    input_data_dict = input_data.dict()
    related_q = fetch_related_questions(input_data_dict["main_keyword"], input_data_dict["number_of_questions"])
    # print(related_q)
    return {"message": "Questions fetched successfully", "data": related_q}


# Endpoint to fetch active articles for a specific user
@app.post("/get_active_articles")
async def get_active_articles(input_data: GetArticlesInput):
    input_data = input_data.dict()
    try:
        # Query MongoDB for active articles by the user's email, sorted by created_at in descending order
        active_articles = await db.articles.find({"user": input_data["email"], "is_active": True}).sort("created_at", -1).to_list(length=None)
        article_out = [ArticleOut(
                id=str(each['_id']),
                user=each['user'],
                title=each["title"],
                created_at=str(each["created_at"]),
                article=each["article"],
                image=each["image"],
                last_modified=str(each["last_modified"]),
                key_words=""
            ) for each in active_articles]
        return article_out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to fetch a single active article by its ID
@app.get("/get_single_article/{article_id}", response_model=ArticleOut)
async def get_single_article(
    article_id: str = Path(..., description="The ID of the article to retrieve"),
    request: Request = None
):
    try:
        article = await db.articles.find_one({"_id": ObjectId(article_id), "is_active": True})
        if article is None:
            raise HTTPException(status_code=404, detail="Article not found")

        # Convert ObjectId to string for JSON serialization
        article["id"] = str(article["_id"])
        del article["_id"]
        
        # Convert datetime objects to strings
        article["created_at"] = article["created_at"].isoformat()
        article["last_modified"] = article["last_modified"].isoformat()
        
        # If image exists and is a full URL, download and save it locally
        if article.get("image") and article["image"].startswith("http"):
            image_response = requests.get(article["image"])
            if image_response.status_code == 200:
                filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(get_image_dir(), filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                
                # Update the database with just the filename
                await db.articles.update_one(
                    {"_id": ObjectId(article_id)},
                    {"$set": {"image": filename}}
                )
                article["image"] = filename

        return article
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to update an article by its ID
@app.put("/update_article/{article_id}")
async def update_article(
    article_id: str = Path(..., description="The ID of the article to update"),
    update_data: ArticleUpdate = Body(..., description="JSON payload containing the updated article content")
):
    # Validate ObjectId
    try:
        # Check if article exists and is active
        existing_article = await db.articles.find_one({"_id": ObjectId(article_id), "is_active": True})
        if not existing_article:
            return JSONResponse(content={"message": "Article not found or is inactive"}, status_code=404)

        # Update the article content and last modified date
        updated_article = await db.articles.update_one(
            {"_id": ObjectId(article_id), "is_active": True},
            {"$set": {"article": update_data.article, "last_modified": datetime.utcnow()}}
        )

        # Check if update was successful
        if updated_article.modified_count == 1:
            return JSONResponse(content={"message": "Article updated successfully"}, status_code=200)
        else:
            return JSONResponse(content={"message": "No changes were made to the article"}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to "delete" an article by marking it as inactive
@app.delete("/delete_article/{article_id}")
async def delete_article(article_id: str = Path(..., description="The ID of the article to delete")):
    # Validate ObjectId
    try:
        # Check if the article exists and is active
        existing_article = await db.articles.find_one({"_id": ObjectId(article_id), "is_active": True})
        if not existing_article:
            return JSONResponse(content={"message": "Article not found or is already inactive"}, status_code=404)

        # Update the is_active field to False
        update_result = await db.articles.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": {"is_active": False}}
        )

        # Check if the update was successful
        if update_result.modified_count == 1:
            return JSONResponse(content={"message": "Article marked as inactive successfully"}, status_code=200)
        else:
            return JSONResponse(content={"message": "No changes were made to the article"}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to serve images
@app.get("/home/charan/Downloads/professional_dev/ver2/be/images/{image_path:path}")
async def get_image(image_path: str):
    image_full_path = os.path.join(get_image_dir(), image_path)
    if not os.path.exists(image_full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    response = FileResponse(image_full_path)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
