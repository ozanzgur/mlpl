import setuptools
import os

pkgname = 'mlpl'

with open("README.md", "r") as fh:
    long_description = fh.read()

script_names = [os.path.join('mlpl', f) for f in os.listdir('mlpl') if ('.py' in f)]
script_names = script_names + [os.path.join('mlpl/pipetools', f) for f in os.listdir('mlpl/pipetools') if ('.py' in f)]

requirements = ["dill>=0.3.1.1", "hyperopt>=0.2.2", "ipython>=7.7.0",
                "jupyter>=1.0.0", "pandas>=0.25.1", "scikit-learn>=0.21.2",
                "scipy>=1.3.0", "tqdm>=4.36.1", "lightgbm>=2.2.3",
                "json5>=0.8.5"]

setuptools.setup(
     name='mlpl',
     version='0.1',
     scripts=script_names,
     author="Ozan Özgür",
     author_email="ozan.zgur@gmail.com",
     description="A data science pipeline tool to speed up data science life cycle.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ozanzgur/mlpl",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent"
         'Topic :: Software Development :: Libraries',
         'Topic :: Software Development :: Libraries :: Python Modules',
         'Intended Audience :: Developers',
     ],
     install_requires=requirements,
     data_files = [('', ['mlpl/count_encoding_model/countencoding_model_ieee', 'mlpl/count_encoding_model/countencoding_model_ieee_scaler'])],
     include_package_data=True,
     #package_data=datafiles
 )