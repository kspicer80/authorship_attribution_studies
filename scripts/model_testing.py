def test_model(text_chunks, reverse_word_index, model, cutoff=0):
    results = []
    print("Analysis: ")
    for test in text_chunks:
        if len(test) > 2:
            print(test)
            predict = model.predict([test])
            if predict[0] > cutoff:
                print("Prediction: " +str(predict[0]))
                results.append(str(predict[0]), reconst_text(test, reverse_word_index))
    return(results, cutoff)

def write_test(results, filename, name):
    with open(filename+.'.txt', 'w', encoding='utf-8') as f:
        f.write("**********TEST ON**************")
        f.write(f"**********{name}**************")
        for result in results:
            f.write(str(result)+'.txt')
