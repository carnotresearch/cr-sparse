# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
import re
import urllib
from urllib.parse import urlparse
import requests
from dataclasses import dataclass

_INITIALIZED = False

CACHE_DIR = ''


def is_valid_url(url):
    import re
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url is not None and regex.search(url)

def _initialize():
    global _INITIALIZED
    if _INITIALIZED:
        return
    # print("Initializing CR-VISION")
    home_dir = Path.home()
    cache_dir = home_dir / '.cr-sparse'
    # Make sure that lib directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    global CACHE_DIR
    CACHE_DIR = cache_dir
    _INITIALIZED = True

_initialize()

def ensure_resource(name, uri=None):
    # technically this line is not required. But it helps in unit test coverage
    _initialize()
    if is_valid_url(name):
        # it seems uri has been passed first
        name, uri = uri, name
    if name is None:
        if uri is None:
            return None
        # let's construct name from uri
        p = urlparse(uri)
        name = p.path.split('/')[-1]

    path = CACHE_DIR  / name
    if path.is_file():
        # It's already downloaded, nothing to do.
        return path
    if uri is None:
        uri = get_uri(name)
    if uri is None:
        # We could not find the download URL
        return None
    r = requests.get(uri, stream=True)
    CHUNK_SIZE = 1024
    print(f"Downloading {name}")
    with path.open('wb') as o:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            o.write(chunk)
    print("Download complete for {}".format(name))
    return path


@dataclass
class _Resource:
    name: str
    uri: str

_KNOWN_RESOURCES = [
    _Resource(name="haarcascade_frontalface_default.xml", 
        uri="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"),
    _Resource(name="lbfmodel.yaml", 
        uri="https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")
]


def get_uri(name):
    for res in _KNOWN_RESOURCES:
        if res.name == name:
            return res.uri
    return None