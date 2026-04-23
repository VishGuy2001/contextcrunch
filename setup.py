from setuptools import setup, find_packages

setup(
    name="contextcrunch",
    version="0.1.0",
    author="Vishnu Sekar",
    description="Token analysis and compression for AI conversations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishGuy2001/contextcrunch",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tiktoken>=0.7.0",
        "sentence-transformers>=3.0.1",
        "scikit-learn>=1.5.0",
        "numpy>=1.26.4",
        "groq>=0.9.0",
        "google-generativeai>=0.7.2",
    ],
    extras_require={
        "files": ["pymupdf>=1.24.5","python-docx>=1.1.2","python-pptx>=0.6.23","openpyxl>=3.1.5","Pillow>=10.4.0"],
        "full": ["pymupdf>=1.24.5","python-docx>=1.1.2","python-pptx>=0.6.23","openpyxl>=3.1.5","Pillow>=10.4.0","fastapi>=0.115.0","uvicorn>=0.30.0","python-multipart>=0.0.9","python-dotenv>=1.0.1"],
    },
    classifiers=["Programming Language :: Python :: 3","License :: OSI Approved :: MIT License","Topic :: Scientific/Engineering :: Artificial Intelligence"],
)
