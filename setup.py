from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Zhou_XJTLU_task1',
    version='0.1.0',
    description='Baseline Inference package',
    author='Ziyang Zhou',
    author_email="ziyang.zhou22@student.xjtu.edu.cn",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Zhou_XJTLU_task1': ["resources/*.wav", 'ckpts/*.ckpt'],
    },
    python_requires='>=3.10',
)
