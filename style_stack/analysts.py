import faiss
import numpy as np
from sklearn import preprocessing


from utils import load_image, get_image_paths


class Analyst:
    def __init__(self, V_dict):
        self.classes = list(V_dict)
        self.V_dict = V_dict
        self._V_dict = V_dict  # save a copy that won't be

    @property
    def V_norm_dict(self):
        d = {}
        for class_, V in self.V_dict.items():
            V_norm = preprocessing.normalize(V, norm='l2')
            d[class_] = V_norm
        return d

    @property
    def V_norm_mean_dict(self):
        d = {}
        for class_, V_norm in self.V_norm_dict.items():
            V_norm_mean = np.mean(V_norm, axis=0)
            d[class_] = V_norm_mean
        return d

    @property
    def V_mean_norm_dict(self):
        d = {}
        for class_, V in self.V_dict.items():
            V_mean = np.mean(V, axis=0)
            V_mean = np.expand_dims(V_mean, axis=0)
            V_mean_norm = preprocessing.normalize(V_mean, norm='l2')
            V_mean_norm = V_mean_norm.flatten()
            d[class_] = V_mean_norm
        return d

    @property
    def V_norm_mean_res_dict(self):
        V_norm_mean_avg = np.mean(np.stack(list(self.V_norm_mean_dict.values())), axis=0)
        d = {class_: mean - V_norm_mean_avg for class_, mean in self.V_norm_mean_dict.items()}
        return d

    @property
    def V_mean_norm_res_dict(self):
        V_mean_norm_avg = np.mean(np.stack(list(self.V_mean_norm_dict.values())), axis=0)
        d = {class_: mean - V_mean_norm_avg for class_, mean in self.V_mean_norm_dict.items()}
        return d

    @property
    def silhouette_score(self):
        V_list = []
        labels = []
        for class_, class_V in self.V_dict.items():
            labels += len(class_V) * [class_]
            V_list.append(class_V)
        V = np.vstack(V_list)
        score = silhouette_score(V, labels, metric='cosine')
        return score


class DenseAnalyst(Analyst):
    @classmethod
    def build(cls, img_by_class, extractor):
        V_dict = cls._build_V_dict(img_by_class, extractor)
        inst = cls(V_dict)
        return inst

    @staticmethod
    def _build_V_dict(img_by_class, extractor):
        V_dict = dict()
        for class_, paths in img_by_class.items():
            x_list = [load_image(path, extractor.input_shape[1:3])[1] for path in paths]
            X = np.vstack(x_list)
            V = extractor.predict(X)
            V_dict[class_] = V
        return V_dict


class AnalystExtras(Analyst):
    @property
    def avg_intraclass_dist_dict(self):
        d = {}
        for class_, V_norm in self.V_norm_dict.items():
            dist_arr = np.array([])
            dim = V_norm.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(V_norm))
            for i, v in enumerate(V_norm[:-1]):
                ref_idx = i + 1
                V_ref = V_norm[ref_idx:]
                V_ref_indices = list(range(ref_idx, ref_idx + len(V_ref)))
                v = np.expand_dims(v, axis=0)
                labels_iter_range = list(range(1, len(V_ref) + 1))
                labels = np.array([V_ref_indices, labels_iter_range])
                distances = np.empty((1, len(V_ref)), dtype='float32')
                index.compute_distance_subset(
                    1, faiss.swig_ptr(v), len(V_ref),
                    faiss.swig_ptr(distances), faiss.swig_ptr(labels))
                distances = distances.flatten()
                dist_arr = np.append(dist_arr, distances)
            print(f'intraclass distances: {dist_arr}')
            avg_dist = np.mean(dist_arr)
            d[class_] = avg_dist
        return d

    @property
    def new_avg_interclass_dist_dict(self):
        d = {}
        class_indices = {}
        i = 0
        V_norm = np.vstack(self.V_norm_dict.values())
        dim = V_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(V_norm))
        for class_, V_norm_class in self.V_norm_dict.items():
            class_indices[class_] = list(range(i, len(V_norm_class)))
            distances, closest_indices = index.search(V_norm_class, len(V_norm))
            dist_avg = np.mean(distances)
            d[class_] = dist_avg
        return d

    @property
    def mean_avg_intraclass_dist(self):
        return np.mean(self.avg_intraclass_dist_dict.values())

    @property
    def avg_interclass_centroid_dist_dict(self):
        # Should I use norm_mean or mean_norm for centroid?
        d = {}
        class_list = []
        v_list = []
        # TODO: TEMPORARILY CHANGED TO V_mean_norm
        for class_, v in self.V_mean_norm_dict.items():
            class_list.append(class_)
            v_list.append(v)
        V_norm_mean = np.stack(v_list)

        dim = V_norm_mean.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(V_norm_mean))

        for i, v in enumerate(V_norm_mean):
            V_ref_indices = list(chain(range(0, i), range(i + 1, len(V_norm_mean))))
            v = np.expand_dims(v, axis=0)
            labels_iter_range = list(range(1, len(V_norm_mean)))
            labels = np.array([list(V_ref_indices), labels_iter_range])
            distances = np.empty((1, len(V_norm_mean) - 1), dtype='float32')
            index.compute_distance_subset(
                1, faiss.swig_ptr(v), len(V_norm_mean),
                faiss.swig_ptr(distances), faiss.swig_ptr(labels))
            distances = distances.flatten()
            print(f'centroid distances: {distances}')
            avg_dist = np.mean(distances)
            d[class_list[i]] = avg_dist
        return d
