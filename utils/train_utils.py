import torch.optim as optim


def build_optim(args, params):
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = _build_opt(args['opt'], filter_fn, args['lr'], args['weight_decay'])
    if 'opt_scheduler' in args:
        scheduler = _build_scheduler(args['opt_scheduler'], optimizer, args)
    else:
        scheduler = None
        print('no scheduler')
    return optimizer, scheduler


def _build_opt(opt: str, params, lr, weight_decay):
    opt = opt.lower()
    if opt == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.95, weight_decay=weight_decay)
    raise NotImplementedError


def _build_scheduler(opt_scheduler: str, optimizer, args):
    opt_scheduler = opt_scheduler.lower()
    if opt_scheduler == 'none':
        return None
    if opt_scheduler == 'multi_step':
        return optim.lr_scheduler.MultiStepLR(optimizer,
                                              milestones=args['milestones'],
                                              gamma=args['gamma'])
    if opt_scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=args['opt_decay_step'], gamma=args['opt_decay_rate'])
    if opt_scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['opt_restart'])
    raise NotImplementedError
