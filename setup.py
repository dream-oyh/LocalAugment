import setuptools

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()
setuptools.setup(
    name="local-augment",
    version="0.1.1",
    license="MIT license",
    author="TayDream",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="19859860010@163.com",
    description="A package for local data augmentation in deep learning",
    url="https://github.com/dream-oyh/LocalAugment",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
