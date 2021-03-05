# -*- coding: utf-8 -*-
import os
try:
    import cPickle
except ImportError:
    import pickle as cPickle

import fire
import tqdm

from utils import iterate_data_files


class MostPopular(object):
    topn = 100

    def __init__(self, from_dtm, to_dtm, tmp_dir='./tmp/'):
        self.from_dtm = str(from_dtm)
        self.to_dtm = str(to_dtm)
        self.tmp_dir = tmp_dir

    def _get_model_path(self) -> str:
        model_path = os.path.join(self.tmp_dir, 'mp.model.%s.%s' % (self.from_dtm, self.to_dtm))
        return model_path

    def _build_model(self):
        model_path = self._get_model_path()
        if os.path.isfile(model_path):
            return

        freq = {}
        print('building model..')
        for path, _ in tqdm.tqdm(iterate_data_files(self.from_dtm, self.to_dtm),
                                 mininterval=1):
            for line in open(path):
                seen = line.strip().split()[1:] # 어떤 유저가 조회한 글 목록
                for s in seen:
                    freq[s] = freq.get(s, 0) + 1 # 등장한 글 하나씩 빈도 추가
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True) # [(글, 빈도), ...] 형태의 빈도 리스트
        open(model_path, 'wb').write(cPickle.dumps(freq, 2))
        print('model built')

    def _get_model(self):
        model_path = self._get_model_path()
        self._build_model()
        ret = cPickle.load(open(model_path, 'rb'))
        return ret

    def _get_seens(self, users):
        set_users = set(users)
        seens = {}
        for path, _ in tqdm.tqdm(iterate_data_files(self.from_dtm, self.to_dtm),
                                 mininterval=1):
            for line in open(path):
                tkns = line.strip().split()
                userid, seen = tkns[0], tkns[1:]
                if userid not in set_users:
                    continue
                seens[userid] = seen
        return seens

    def recommend(self, userlist_path, out_path):
        mp = self._get_model()
        mp = [a for a, _ in mp] # 글 목록

        with open(out_path, 'w') as fout:
            users = [u.strip() for u in open(userlist_path)]
            seens = self._get_seens(users) # 유저 목록의 각 유저들이 조회한 글을 딕셔너리 형태로 보여줌
            for user in users:
                seen = set(seens.get(user, []))
                recs = mp[:self.topn + len(seen)]
                sz = len(recs)
                recs = [r for r in recs if r not in seens]
                if sz != len(recs):
                    print(sz, len(recs))
                fout.write('%s %s\n' % (user, ' '.join(recs[:self.topn])))

if __name__ == '__main__':
    fire.Fire(MostPopular)
