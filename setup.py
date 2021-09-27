import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trackun",
    version="0.0.1",
    author="The-Anh Vu-Le",
    author_email="narubees@gmail.com",
    description="A Python package for (multiple) object tracking using recursive Bayesian filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vltanh/trackun",
    project_urls={
        "Bug Tracker": "https://github.com/vltanh/trackun/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'numba'
    ],
)
