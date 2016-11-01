import artm

# batch_vectorizer = artm.BatchVectorizer(
#                                            data_format='bow_uci',
#                                            data_path="corpus",
#                                            collection_name='uci_corpus_3k_10themes',
#                                            target_folder='artm_batches')

batch_vectorizer = artm.BatchVectorizer(data_path='artm_batches',
                                         data_format='batches')
# 129 total
T = 10
model = artm.ARTM(num_topics=T, topic_names=["sbj"+str(i) for i in range(T)], class_ids={"@default_class": 1})

model.scores.add(artm.PerplexityScore(name='my_fisrt_perplexity_score',
                                      use_unigram_document_model=False,
                                      dictionary=batch_vectorizer.dictionary))

model.scores.add(artm.TopTokensScore(name="top_words", num_tokens=15, class_id="@default_class"))

model.initialize(batch_vectorizer.dictionary)

model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)


print model.score_tracker['my_fisrt_perplexity_score'].value
print model.score_tracker['top_words'].tokens

# for topic_name in model.topic_names:
#     print topic_name + ': ',
#     for word in model.score_tracker["top_words"].last_topic_info[topic_name].tokens:
#         print word,
#     print

