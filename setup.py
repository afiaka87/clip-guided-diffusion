import os

import pkg_resources
from setuptools import setup, find_packages
from pathlib import Path

if __name__ == "__main__":

    # Read description from README
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="cgd-pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        description=long_description.split("\n")[0],
        url="https://github.com/afiaka87/clip-guided-diffusion",
        keywords = [
            'text to image',
            'openai clip',
            'guided diffusion',
            'image processing',
            'pytorch'
        ],
        py_modules=["cgd"],
        version="0.2.5",
        author="Katherine Crowson, Clay Mullis",
        entry_points={
            'console_scripts': ['cgd = cgd.cgd:main', ],
        },
        packages=find_packages(exclude=["tests*"]),
        install_requires=[
            str(r)
            for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
            )
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        # extras_require={'dev': ['']},
    )
