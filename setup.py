from setuptools import setup, find_packages
import itertools

# Parse the version from the main __init__.py
with open('camtraproc/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

extra_reqs = {'docs': ['sphinx',
                       'sphinx-rtd-theme',
                       'sphinxcontrib-programoutput'],
              's3': ['boto3'],
              'tensorflow': ['tensorflow'],
              'others': ['pandas',
                        'numpy']}

extra_reqs['all'] = list(set(itertools.chain(*extra_reqs.values())))

setup(name='camtraproc',
      version=version,
      description=u"Detection and classification of camera trap images",
      classifiers=[],
      keywords='Camera trap, detection, classification',
      author=u"ixime",
      author_email='',
      url='https://github.com/CONABIO/detector-clasificador-especies',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'python-dotenv',
          'requests',
          'sklearn',
          'scikit-image',
          'os'],
      entry_points={'console_scripts': [
          'camtraproc = camtraproc:main',
      ]},
      include_package_data=True,
      test_suite="tests",
      extras_require=extra_reqs)
