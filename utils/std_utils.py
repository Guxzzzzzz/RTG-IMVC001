import numpy as np


def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 3:
        acc = np.asarray(arg[0], dtype=float)
        nmi = np.asarray(arg[1], dtype=float)
        ari = np.asarray(arg[2], dtype=float)
        means = 100.0 * np.asarray((acc.mean(), nmi.mean(), ari.mean()))
        stds = 100.0 * np.asarray((acc.std(), nmi.std(), ari.std()))
        width = 72
        logger.info('=' * width)
        logger.info('FINAL CLUSTERING RESULTS | seeds={}'.format(acc.size))
        logger.info('-' * width)
        logger.info('  ACC  {:>7.2f} +/- {:<7.2f}'.format(means[0], stds[0]))
        logger.info('  NMI  {:>7.2f} +/- {:<7.2f}'.format(means[1], stds[1]))
        logger.info('  ARI  {:>7.2f} +/- {:<7.2f}'.format(means[2], stds[2]))
        if acc.size == 1:
            logger.info('  Note: one seed only; the reported std is not informative.')
        logger.info('-' * width)
        logger.info('  Per-seed ACC: {}'.format(
            ', '.join('{:.2f}'.format(100.0 * value) for value in acc)
        ))
        logger.info('  Per-seed NMI: {}'.format(
            ', '.join('{:.2f}'.format(100.0 * value) for value in nmi)
        ))
        logger.info('  Per-seed ARI: {}'.format(
            ', '.join('{:.2f}'.format(100.0 * value) for value in ari)
        ))
        logger.info('=' * width)
        logger.info(
            'MACHINE_READABLE: '
            'ACC={:.2f},{:.2f};NMI={:.2f},{:.2f};ARI={:.2f},{:.2f};'.format(
                means[0], stds[0], means[1], stds[1], means[2], stds[2]
            )
        )
        return tuple(round(float(value), 2) for value in means)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)


def cal_HAR(logger, *arg):
    """ print classification results for HAR """
    if len(arg) == 5:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        logger.info(arg[3])
        logger.info(arg[4])
        output = """ 
                     RGB {:.2f} std {:.2f}
                     Depth {:.2f} std {:.2f} 
                     RGB+D {:.2f} std {:.2f}
                     onlyrgb {:.2f} std {:.2f}
                     onlydepth {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
                                                           np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
                                                           np.mean(arg[3]) * 100, np.std(arg[3]) * 100,
                                                           np.mean(arg[4]) * 100, np.std(arg[4]) * 100)

        logger.info(output)
    return


def cal_classify(logger, *arg):
    """ print classification results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f} 
                     F-measure {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100,
                                                           np.std(arg[1]) * 100, np.mean(arg[2]) * 100,
                                                           np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)
    return
