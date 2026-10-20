"""Per-beam Monte Carlo means, then a pointwise maximum (never batch maxima)."""
import numpy as np


class CoverageAccumulator:
    def __init__(self,beam_count):
        self.counts=np.zeros(beam_count,dtype=np.int64)
        self.means=None

    def next_beam(self,budget):
        i=int(np.argmin(self.counts))
        return i if self.counts[i]<budget else None

    def add(self,index,values,samples):
        if self.means is None:
            self.means=np.zeros((len(self.counts),*values.shape),dtype=np.float32)
        total=int(self.counts[index])+samples
        self.means[index]+=(values-self.means[index])*(samples/total)
        self.counts[index]=total

    def result(self):
        visited=self.counts>0
        if not visited.any():return None,None
        indices=np.flatnonzero(visited)
        # Array shape [beam, tx, row, col]; demo supports one active TX.
        observed=self.means[visited]
        winner=indices[np.argmax(observed,axis=0)]
        values=np.max(observed,axis=0)
        winner=np.where(values>0,winner,-1)
        return values,winner
