import os
import sys
import glob
from functools import partial
from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from crow import run_feature_processing_pipeline,compute_crow_channel_weight,compute_crow_spatial_weight, apply_crow_aggregation,apply_ucrow_aggregation, apply_max_aggregation,normalize,save_spatial_weights_as_jpg

def get_nn(x, data, k=None):
    if k is None:
        k = len(data)

    dists = ((x-data)**2).sum(axis=1)
    idx = np.argsort(dists)
    dists = dists[idx]

    return idx[:k], dists[:k]

def simple_query_expansion(Q, data, inds, top_k):
    Q += data[inds[:top_k], :].sum(axis=0)
    return normalize(Q)
def load_features(feature_dir, verbose=True):
    if type(feature_dir) == str:
        feature_dir = [feature_dir]
    for  directory in feature_dir:
        for i, f in enumerate(os.listdir(directory)):
            name = os.path.splitext(f)[0]

            if verbose and not i % 100:
                sys.stdout.write('\rProcessing file %i' % i)
                sys.stdout.flush()

            X = np.load(os.path.join(directory, f))
            print(X.shape)

            yield X, name
    sys.stdout.write('\n')
    sys.stdout.flush()

def load_and_aggregation_features(feature_dir, agg_fn):
    print('Loading features %s ..' % str(feature_dir))
    features = []
    names = []
    for X , name in load_features(feature_dir):
        names.append(name)
        X = agg_fn(X)
        #X = normalize(X)
        features.append(X)
    return features, names
def get_ap(inds, dists, query_name, index_names, groundtruth_dir, ranked_dir='rank'):
    if ranked_dir is not None:
        if not os.path.exists(ranked_dir):
            os.makedirs(ranked_dir)
        rank_file = os.path.join(ranked_dir, '%s.txt' % query_name)
        print('rank_fle=',rank_file)
        f = open(rank_file, 'w')
    else:
        f = NamedTemporaryFile(delete=False)
        rank_file = f.name
    for i in inds:
       # print('i=', i)
        f.writelines([(index_names[i]+'\n')])
    f.close()

    groundtruth_prefix = os.path.join(groundtruth_dir,query_name)
    print('groundtruth_prefix, rank_file = ', groundtruth_prefix, rank_file)
    cmd = 'compute_ap %s %s ' % (groundtruth_prefix, rank_file)
    ap = os.popen(cmd).read()
    print('ap=',ap)
    if ranked_dir is None :
        os.remove(rank_file)
    apf = float(ap.strip())
    print('apf=', apf)
    if apf < 0.5:
        path = groundtruth_prefix+'_query.txt'
        img_name, x, y, w, h = open(path).read().strip().split(' ')
        img_name = img_name.replace('oxc1_', '')

        img = io.imread(os.path.join('data',img_name)+'.jpg')
        io.imsave(os.path.join('badimage',img_name)+'.jpg',img)
    return float(ap.strip())

def fit_whitening(whiten_features, agg_fn, d):
    data, _ = load_and_aggregation_features(whiten_features, agg_fn)
    print('Fitting PCA/whitening with d=%d on %s' %(d, whiten_features))
    _, whiten_params = run_feature_processing_pipeline(data, d=d)

    return whiten_params

def run_eval(queries_dir, groundtruth_dir, index_features, whiten_params, out_dir, agg_fn, qe_fn=None):
    data,image_names = load_and_aggregation_features(index_features, agg_fn)
    data = np.array(data)
    #print('whitten2=', whitening_params)
    #print(data.shape)
    data, _ = run_feature_processing_pipeline(np.vstack(data), params=whiten_params)

    aps = []
   # print("query_dir=", queries_dir)
    print('data.',data.shape)
    for Q, query_name in load_features(queries_dir):
        Q = agg_fn(Q)
        #print('whitten3=', whitening_params)
        Q, _ = run_feature_processing_pipeline(Q, params=whiten_params)
        #print(Q.shape)
       # Q = normalize(Q)
        inds, dists =get_nn(Q, data)
      #  print('inds=', inds)
       # print('dists', dists)


        if qe_fn is not None:
            Q= qe_fn(Q, data, inds)
            inds, dists = get_nn(Q, data)
        print(inds.shape)
        print(len(image_names))
        ap = get_ap(inds, dists, query_name, image_names, groundtruth_dir, out_dir)

        aps.append(ap)

    return np.array(aps).mean()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--wt', dest='weighting', type=str, default='crow')
    parser.add_argument('--index_features', dest='index_features',type=str, default='features')
    parser.add_argument('--whiten_features', dest = 'whiten_features', type=str, default='features')

    parser.add_argument('--queries', dest='queries', type=str, default='pool5_queries')

    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='groundtruth',
                        help='directory containing groundtruth files')
    parser.add_argument('--d', dest='d', type=int, default=256, help='dimension of final feature')
    parser.add_argument('--out', dest='out', type=str, default='rank', help='optional path to save ranked output')
    parser.add_argument('--qe', dest='qe', type=int, default=10,
                        help='perform query expansion with this many top results')
    args = parser.parse_args()
    if args.weighting == 'crow':
        agg_fn = apply_crow_aggregation
    elif args =='max':
        agg_fn = apply_max_aggregation
    else:
        agg_fn = apply_ucrow_aggregation
    if args.qe > 0:
        qe_fn = partial(simple_query_expansion, top_k=args.qe)
    else:
        qe_fn = None

    whitening_params = fit_whitening(args.whiten_features, agg_fn, args.d)
   # print('whitten1=',whitening_params)
    #compute aggregated features and run the evaluation
    mAP = run_eval(args.queries, args.groundtruth, args.index_features,whitening_params, args.out, agg_fn, qe_fn)
    print( 'mAP: %f' % mAP)

    exit(0)