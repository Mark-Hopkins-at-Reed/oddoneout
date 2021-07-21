"""
download.py

Downloads relevant Wikipedia dumps (enwiki and enwikibooks) 
and converts them to text files using the WikiUtils GitHub repo.
The files downloaded and extracted here are utilized in wiki_demo.py,
which converts the text files to ontologies.
"""

import requests # requests.get, for downloading wiki dumps
import subprocess # subprocess.Popen, for running the extraction commands
from pathlib import Path # for checking if a file already exists in pwd

def analyze_dump_filename(filename):
    (_, dump_date, filetype) = filename[:-len('.sql.gz')].split('-')
    return dump_date, filetype


def download(dumps_to_download, wikidumps_url = "https://dumps.wikimedia.org"):
    for dump in dumps_to_download:
        # First check if dump is already downloaded
        if not Path(dump).is_file():
                        
            # Form the URL of the dump to download and issue request to it
            wiki = dump[:dump.find('-')]
            dump_date, _ = analyze_dump_filename(dump)
            dump_url = wikidumps_url + "/" + wiki + "/" + dump_date + "/" + dump
            dump_contents = requests.get(dump_url, stream=True)
            
            # Write dump to file
            with open(dump, "wb") as dump_file:
                for chunk in dump_contents.iter_content(chunk_size=1024):
                    if chunk:
                        dump_file.write(chunk)


def extract(dumps_to_download):
    extracted_files = [filename[:-len('.sql.gz')] for filename in dumps_to_download]
    for i in range(len(extracted_files)):
        output_file = extracted_files[i]
        dump = dumps_to_download[i]
        _, filetype = analyze_dump_filename(dump)
        # Check if files have already been extracted
        if not Path(extracted_files[i]).is_file():
            # Extract into text file using WikiUtils repo
            proc = subprocess.Popen(["python", "WikiUtils/parse_mysqldump.py",
                                     dump, filetype, output_file])
            proc.wait()


if __name__ == "__main__":
    # Initialize dump variables
    my_dump_date = "20201020"
    dumps_to_download = [
        "enwiki-" + my_dump_date + "-page.sql.gz",
        "enwiki-" + my_dump_date + "-categorylinks.sql.gz",
        "enwikibooks-" + my_dump_date + "-page.sql.gz",
        "enwikibooks-" + my_dump_date + "-categorylinks.sql.gz"]
    download(dumps_to_download)
    extract(dumps_to_download)