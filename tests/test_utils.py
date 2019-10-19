from tree.optimized_train import utils


def test_booster_text():
    text = '''0:[f10<0.25] yes=1,no=2,missing=1
	1:[f100<0.0500000007] yes=3,no=4,missing=3
		3:leaf=-0.229934558
		4:leaf=-0.495003372'''
    assert utils.increase_leaves_booster_text(text, 0.5) == '''0:[f10<0.25] yes=1,no=2,missing=1
	1:[f100<0.0500000007] yes=3,no=4,missing=3
		3:leaf=0.270065442
		4:leaf=0.004996628000000003'''
