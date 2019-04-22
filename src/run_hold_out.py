if __name__ == "__main__":
    import sys
    import json
    sys.path.append('./src')

    from libs.EvaluateModels import EvaluateModels

    params = json.load(open('./src/params/submit_11.json'))

    test = EvaluateModels(params)
    test.run_hold_out()
