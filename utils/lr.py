def poly_lr_scheduler(optimizer, epoch, lr_decay_rate=0.1, decay_epoch=80):
    if epoch!=0 and epoch % decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate

def poly_lr_scheduler2(optimizer, epoch, lr, maintain_epoch=0, decay_epoch=100):
    lr_l = 1.0 - (max(0, epoch - maintain_epoch) / float(decay_epoch + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr*lr_l