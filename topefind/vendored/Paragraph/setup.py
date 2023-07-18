from setuptools import setup, find_packages


setup(
    name='Paragraph',
    version='1.0.0',
    description='Paratope prediction using Equivariant Graph Neural Networks',
    license='BSD 3-clause license',
    maintainer='Lewis Chinery',
    long_description_content_type='text/markdown',
    maintainer_email='lewis.chinery@dtc.ox.ac.uk',
    include_package_data=True,
    package_data={'': ['trained_model/*', 'example/*', 'example/pdbs/*']},
    packages=find_packages(include=('Paragraph', 'Paragraph.*')),
    entry_points={'console_scripts': ['Paragraph=Paragraph.command_line:main']},
    install_requires=[
        'numpy',
        'pandas',
        'einops>=0.4',
        'scipy>=1.8',
        'torch>=1.11',
        'tqdm'
    ],
)
