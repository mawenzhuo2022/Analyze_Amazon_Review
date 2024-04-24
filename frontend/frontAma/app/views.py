import os
import csv
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from dotenv import load_dotenv
from openai import OpenAI
from django.http import HttpRequest, HttpResponse
from django.template import loader
from .backend.src.analyze import analyze_main
import sys
print(sys.path)



# Create your views here.
def index(_: HttpRequest):
    template = loader.get_template('../templates/index.html')
    return HttpResponse(template.render())

def summary(_: HttpRequest):
    template = loader.get_template('../templates/summary.html')
    return HttpResponse(template.render())

def scoring(_: HttpRequest):
    template = loader.get_template('../templates/scoring.html')
    return HttpResponse(template.render())

def pioritize(_: HttpRequest):
    template = loader.get_template('../templates/pioritize.html')
    return HttpResponse(template.render())

def pioritizeD(_: HttpRequest):
    template = loader.get_template('../templates/pioritizeD.html')
    return HttpResponse(template.render())

"""backend function for summarization """
# get API KEY
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# preprocess the csv file
def csv_content_to_string(file):
    aggregated_reviews = {}
    # Read the content of the file and decode it to utf-8
    file_content = file.read().decode('utf-8')
    reader = csv.DictReader(file_content.splitlines())
    for row in reader:
        product_name = row['productName']
        review_text = row['reviewText']
        if product_name in aggregated_reviews:
            aggregated_reviews[product_name].append(review_text)
        else:
            aggregated_reviews[product_name] = [review_text]

    formatted_reviews = []
    for product_name, reviews in aggregated_reviews.items():
        reviews_str = "\n".join(f"Review: {review}" for review in reviews)
        formatted_review = f"Product Name: {product_name}\n{reviews_str}"
        formatted_reviews.append(formatted_review)
    return formatted_reviews

class UploadCSV(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        file_obj = request.FILES['csvFile']
        reviews = csv_content_to_string(file_obj)
        if reviews:
            system_prompt = f"You are a effective summarizer and extract the key general aspects covered in the product reviews. And you're responsible to remove the overlapping redundant aspects."
            user_prompt = f"""Here is a list of Amazon reviews about Product Name. Please summarize different general aspects 
                    covered in it such as quality, price, appearence. And express in the neutral words.
                    Example Review: I found the QuickSnap to be exceptionally durable; it survived a drop 
                    during a mountain hike without a scratch. 
                    It's a bit heavier than I expected, but the image quality is so superior that it's worth the extra weight.
                    Example Result: [Durability, Image Quality, Weight]
                    Please concisely summarize the general aspects without duplicate items and return as a list in the format '[Aspect 1, Aspect2, ...].f{reviews[0]} :"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature = 0.7 # calla to do 
                )
                return Response({"summary": response.choices[0].message.content})
            except Exception as e:
                return Response({"error": str(e)}, status=400)
        return Response({"error": "No reviews parsed"}, status=400)

class ScoreReview(APIView):
    parser_classes = [JSONParser]

    def post(self, request, *args, **kwargs):
        data = request.data
        aspects = data.get('aspects')
        review = data.get('review')

        if not aspects or not review:
            return Response({"error": "Missing aspects or review"}, status=400)

        system_prompt = "You are an effective review evaluator."
        user_prompt = f"""
        Here is an Amazon product review. Please read through the review and evaluate whether 
        each individual aspect in the aspects list is covered. If so, rate the aspect from negative, neutral, or positive.
        Please only return mentioned aspect without duplicate aspects.
        Aspect List: {aspects}
        Review: {review}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 0.9
            )
            return Response({"summary": response.choices[0].message.content}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class Pioritize(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        try:
            file_obj = request.FILES['csvFile']
            result = analyze_main.main(file_obj)  # 调用 main 函数并获取结果
            print('success')
            return Response({'summary' : result}, status=200)  # 返回处理结果
        except Exception as e:
            return Response({'error' : str(e)}, status=500)