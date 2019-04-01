from setuptools import setup

setup(name='hyptorch',
      version='1.0.0',
      description=('Supplementary code for Hyperbolic Image Embeddings.'),
      url='None',
      author='Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Victor Lempitsky, Ivan Oseledets',
      author_email='khrulkov.v@gmail.com',
      license='MIT',
      packages=['hyptorch'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)