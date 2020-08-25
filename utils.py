def decode(predictions, encoder):
    results = []
    for prediction in predictions:
        string = ''
        for p in prediction:
            if p == -1:
                string += '$'
            else:
                string += encoder.inverse_transform([p])[0]
        results.append(string)

    return results
