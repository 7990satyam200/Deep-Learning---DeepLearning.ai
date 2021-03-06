def read_glove_vecs(glove_file):
    with open(glove_file, 'r+', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            #line = line.strip().split()
            line = line.decode('utf-8').strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
