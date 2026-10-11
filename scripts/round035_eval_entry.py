import sys
sys.path[:0]=['AgenticArxiv','scripts']
from round035_literal_error_log import install
install()
from run_round034_eval import main
if __name__=='__main__':main()
