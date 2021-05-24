import numpy as np
import torch
import torch.nn.functional as f
from scipy.optimize import differential_evolution


def perturb_image(xs, img, preproc):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)
    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)
        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - preproc['mean'][0]) / preproc['std'][0]
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - preproc['mean'][1]) / preproc['std'][1]
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - preproc['mean'][2]) / preproc['std'][2]
        count += 1
    return imgs


def predict_classes(xs, img, target_class, net, preproc, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone(), preproc)
    predictions = f.softmax(net(imgs_perturbed), dim=1).data.cpu().numpy()[:, target_class]
    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, net, preproc, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img.clone(), preproc)
    confidence = f.softmax(net(attack_image), dim=1).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)
    if (verbose):
        print("Confidence: {}".format(confidence[target_class]))
    if (targeted_attack and predicted_class == target_class) or\
            (not targeted_attack and predicted_class != target_class):
        return True


def attack(img, label, net, preproc, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    targeted_attack = target is not None
    target_class = target if targeted_attack else label
    bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels
    popmul = int(max(1, popsize / len(bounds)))

    def predict_fn(xs):
        return predict_classes(xs, img, target_class, net, preproc, target is None)

    def callback_fn(x, convergence):
        return attack_success(x, img, target_class, net, preproc, targeted_attack, verbose)

    inits = np.zeros([popmul * len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random() * 32
            init[i*5+1] = np.random.random() * 32
            init[i*5+2] = np.random.normal(128, 127)
            init[i*5+3] = np.random.normal(128, 127)
            init[i*5+4] = np.random.normal(128, 127)

    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul, callback=callback_fn, init=inits, polish=False)
    # attack_result = differential_evolution(
    #     predict_fn, bounds, maxiter=maxiter, popsize=popmul, atol=-1, callback=callback_fn, init=inits, polish=False)

    attack_image = perturb_image(attack_result.x, img, preproc)
    predicted_probs = f.softmax(net(attack_image), dim=1).data.cpu().numpy()[0]
    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_all(net, loader, preproc, device, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):
    correct = 0
    success = 0
    for batch_idx, (img, target) in enumerate(loader):
        img = img.to(device)
        prior_probs = f.softmax(net(img), dim=1)
        _, indices = torch.max(prior_probs, 1)
        if target[0] != indices.data.cpu()[0]:
            continue
        correct += 1
        target = target.numpy()
        targets = [None] if not targeted else range(10)
        for target_class in targets:
            if (targeted):
                if (target_class == target[0]):
                    continue
            flag, x = attack(img, target[0], net, preproc, target_class,
                             pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
            success += flag
            if (targeted):
                success_rate = float(success) / (9 * correct)
            else:
                success_rate = float(success) / correct
            if flag == 1:
                print("Success rate: {} ({}/{}) [{}]".format(success_rate, success, correct, x))
    return success_rate


one_pixel_attack = attack_all
