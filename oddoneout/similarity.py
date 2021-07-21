class SimilarityScore:

    def __call__(self, words):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def is_recognized(self, word):
        raise NotImplementedError('Cannot call this method on abstract class.')

