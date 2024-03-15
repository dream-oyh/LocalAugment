import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LocalAugment",
    version="0.0.1",
    author="TayDream",
    author_email="19859860010@163.com",
    description="A package for local data augmentation in deep learning",
    long_description=long_description,
    long_description_content_type="markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
