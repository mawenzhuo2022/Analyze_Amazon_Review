import re

def main():
    # 使用正则表达式从第一行提取被三个方括号包围的产品名称
    data = """
    [[[iphone7 (refurbished)]]]

    1. "great" (Coefficient: 0.373546):
    When customers perceive a product as "great," it tends to have a significant positive impact on their satisfaction and ratings. This feature suggests that the product excels in various aspects, such as performance, design, or overall user experience. Customers are likely to appreciate and value a product that is described as "great," leading to higher ratings.

    2. "good" (Coefficient: 0.360725):
    Similar to "great," the feature "good" also has a strong positive impact on customer satisfaction and ratings. When customers find a product to be "good," it implies that the product meets their expectations and performs well. This positive perception often translates into higher ratings as customers are generally satisfied with their purchase.

    3. "new" (Coefficient: 0.142310):
    The presence of the feature "new" suggests that the product is in a pristine condition or has the latest technology. While not as strong as "great" or "good," the perception of a product being "new" still positively influences customer satisfaction and ratings. Customers tend to prefer products that are new over refurbished or older models, leading to slightly higher ratings.

    4. "screen" (Coefficient: 0.061117):
    The feature "screen" highlights the importance of the display quality in influencing customer perceptions and ratings. A good screen quality often enhances the overall user experience, contributing positively to customer satisfaction. Customers appreciate products with high-quality screens, resulting in better ratings for the product.

    5. "use" (Coefficient: 0.055843):
    The feature "use" indicates that the product is user-friendly and easy to operate. When customers find a product easy to use, it enhances their overall satisfaction and leads to better ratings. Products that are intuitive and straightforward tend to receive positive feedback from customers, reflecting in higher ratings.

    6. "work" (Coefficient: 0.040431):
    The feature "work" suggests that the product functions effectively and fulfills its intended purpose. When customers perceive a product as reliable and functional, it positively impacts their satisfaction and ratings. Products that work well and consistently deliver on their promises tend to receive favorable reviews from customers.

    7. "battery" (Coefficient: -0.004944):
    The feature "battery" addresses the battery life of the product, which is a critical aspect for many customers. A negative coefficient implies that a subpar battery performance may slightly impact customer satisfaction and ratings. Customers often value products with long-lasting batteries, and any shortcomings in this area could lead to lower ratings.

    8. "charge" (Coefficient: -0.000367):
    The feature "charge" pertains to the charging functionality of the product. A negligible coefficient suggests that this feature has minimal impact on customer satisfaction and ratings. While efficient charging capabilities are appreciated by customers, it may not significantly influence their overall perception of the product.
    """


    product_name_match = re.search(r'\[\[\[(.*?)\]\]\]', data)
    if product_name_match:
        product_name = product_name_match.group(1)
        print("Product Name:", product_name)  # 打印产品名称

    lines = data.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith(
                '4.') or line.startswith('5.') or line.startswith('6.') or line.startswith('7.') or line.startswith(
            '8.'):
            feature = line.split('"')[1]
            i += 1  # Move to the description line
            description = ""
            while i < len(lines) and not lines[i].strip().startswith('1.') and not lines[i].strip().startswith(
                    '2.') and not lines[i].strip().startswith('3.') and not lines[i].strip().startswith('4.') and not \
                    lines[i].strip().startswith('5.') and not lines[i].strip().startswith('6.') and not lines[
                i].strip().startswith('7.') and not lines[i].strip().startswith('8.'):
                description += lines[i].strip() + " "
                i += 1
            parts = description.strip().rsplit('.', -1)  # Split all sentences on period
            if len(parts) >= 2:  # Ensure there are at least two parts
                second_last_content = parts[-2].strip() + '.'  # Get the second-to-last part
            else:
                second_last_content = description.strip()  # If less than two sentences, take the whole description
            print(f"{feature}: {second_last_content}")  # Print each feature's second-to-last sentence
        else:
            i += 1


# 确保当直接运行此脚本时调用 main 函数
if __name__ == "__main__":
    main()
