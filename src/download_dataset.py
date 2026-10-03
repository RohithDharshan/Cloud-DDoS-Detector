"""
Run this on YOUR OWN LAPTOP (not in a restricted sandbox) to fetch the real
CICIDS2017 dataset. The dev sandbox this project was scaffolded in has no
route to Kaggle / Google Drive / the UNB host, so this step is meant to be
run locally, once, by whoever on the team is assigned the dataset.

Two official/legit sources — pick ONE:

1) Official UNB source (Canadian Institute for Cybersecurity):
   https://www.unb.ca/cic/datasets/ids-2017.html
   -> Click through to the download page, grab "MachineLearningCSV.zip"
      (this is the pre-extracted, CICFlowMeter-processed CSV set — NOT the
      raw pcaps, which you do not need for this project).

2) Kaggle mirror (same data, easier auth if you already have a Kaggle
   account + API token set up: ~/.kaggle/kaggle.json):
   https://www.kaggle.com/datasets/shadman1028/cicids2017-official-flow-feature-csv-files
   kaggle datasets download -d shadman1028/cicids2017-official-flow-feature-csv-files \
     -f "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" -p data/raw
   (note the downloaded filename says "DDos" not "DDoS" — rename it to match
   EXPECTED_PATH below before running preprocess.py)

Either way, you want the file named (or equivalent to):
   Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
This is the single day/file that isolates DDoS attack traffic against
benign traffic, which is exactly the binary classification task this
project targets. Using just this one file (~90-100MB) instead of the full
5-day dataset (~1GB+) keeps training fast and is what the reference paper's
Random Forest benchmark is built on for the DDoS-specific numbers.

Once downloaded, place it at:
   data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv

Then run:
   python src/preprocess.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
"""

import sys
import os

EXPECTED_PATH = os.path.join("data", "raw", "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv")

if __name__ == "__main__":
    print(__doc__)
    if os.path.exists(EXPECTED_PATH):
        size_mb = os.path.getsize(EXPECTED_PATH) / (1024 * 1024)
        print(f"\nFound existing file at {EXPECTED_PATH} ({size_mb:.1f} MB). You're good to go.")
    else:
        print(f"\nNo file found yet at {EXPECTED_PATH}.")
        print("Download it manually from one of the two sources above and place it there.")
        sys.exit(1)
