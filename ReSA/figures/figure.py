import os
from pdf2image import convert_from_path

# 文件夹路径
figures_dir = "figures"

# 遍历 figures/ 目录中的所有 PDF 文件
for filename in os.listdir(figures_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(figures_dir, filename)
        png_name = filename.replace(".pdf", ".png")
        png_path = os.path.join(figures_dir, png_name)

        print(f"Converting {pdf_path} -> {png_path} ...")

        # 转换 PDF -> PNG
        images = convert_from_path(pdf_path, dpi=400, transparent=True)

        # 保存（如果多页，只保存第一页）
        images[0].save(png_path, "PNG")

print("✅ All PDF files converted to high-quality PNG.")
