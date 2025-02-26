# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from evaluation_task.caption.eval.cider_metrics.cider_scorer_pycocoeval import CiderScorer as CiderScorer_pycocoeval
from evaluation_task.caption.eval.cider_metrics.cider_scorer_original import CiderScorer as CiderScorer_original
import pdb

class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, backend='original', test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self._backend = backend

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        if self._backend == 'original':
            cider_scorer = CiderScorer_original(n=self._n, sigma=self._sigma)
        elif self._backend == 'pycocoeval':
            cider_scorer = CiderScorer_pycocoeval(n=self._n, sigma=self._sigma)
        else:
            raise ValueError("Invalid backend: {}".format(self._backend))

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"