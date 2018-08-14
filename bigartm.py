import artm

batch_vectorizer = artm.BatchVectorizer(
                                           data_format='bow_uci',
                                           data_path="corpus",
                                           collection_name='bow',
                                           target_folder='artm_batches')

# batch_vectorizer = artm.BatchVectorizer(data_path='artm_batches',
#                                          data_format='batches')
# 129 total
T = 10
class_priority = {"@default_class": 1, "@ngram_2": 2, "@ngram_3": 6}
model = artm.ARTM(num_topics=T, topic_names=["sbj"+str(i) for i in range(T)], class_ids=class_priority)

model.scores.add(artm.PerplexityScore(name='my_fisrt_perplexity_score',
                                      use_unigram_document_model=False,
                                      dictionary=batch_vectorizer.dictionary))
model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore', class_id="@default_class"))
model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model.scores.add(artm.TopTokensScore(name="top_words", num_tokens=15, class_id="@ngram_3"))

model.initialize(batch_vectorizer.dictionary)

model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)


print model.score_tracker['my_fisrt_perplexity_score'].value

print "SparsityPhiScore: "+str(model.score_tracker["SparsityPhiScore"].last_value)
print "SparsityThetaScore: "+str(model.score_tracker["SparsityThetaScore"].last_value)

for topic_name in model.topic_names:
    print topic_name + ': ',
    tokens = model.score_tracker["top_words"].last_tokens
    for word in tokens[topic_name]:
        print word,
    print


print "\n\n\nWith SmoothSparsePhiRegularizer..."
model.cache_theta = True
model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-100, dictionary=batch_vectorizer.dictionary))
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=25)

print "SparsityPhiScore: "+str(model.score_tracker["SparsityPhiScore"].last_value)
print "SparsityThetaScore: "+str(model.score_tracker["SparsityThetaScore"].last_value)

for topic_name in model.topic_names:
    print topic_name + ': ',
    tokens = model.score_tracker["top_words"].last_tokens
    for word in tokens[topic_name]:
        print word,
    print

model.save("models/3kd_10t_500limit_3n_SparsePhi.model")