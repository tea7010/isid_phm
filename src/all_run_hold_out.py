if __name__ == "__main__":
    import sys
    import json
    sys.path.append('./src')

    from libs.EvaluateModels import EvaluateModels
    json_list = ['./src/params/submit_5.json',
                 './src/params/submit_6.json',
                 './src/params/submit_7.json',
                 './src/params/submit_9.json',
                 './src/params/submit_10.json',
                 './src/params/submit_11.json']

    for json_i in json_list:
        params = json.load(open(json_i))

        test = EvaluateModels(params)
        test.run_hold_out()
