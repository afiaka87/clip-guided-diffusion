from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.nn import functional as F
from torch import clip_, nn
import torch
import clip
import argparse
import time
import os
import re
import sys
from pathlib import Path

from guided_diffusion.nn import checkpoint
from PIL import Image

from torch_util import (spherical_dist_loss, tv_loss)
from util import fetch, load_guided_diffusion


IMAGENET_CLASSES = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop",
                    "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


sys.path.append("./guided-diffusion")


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, num_cutouts, cutout_size_power=1.0, augment_list=[]):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.augs = nn.Sequential(*augment_list)

    def forward(self, input):
        side_x, side_y = input.shape[2:4]
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow *
                (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, side_y - size + 1, ())
            offsety = torch.randint(0, side_x - size + 1, ())
            cutout = input[:, :, offsety: offsety +
                           size, offsetx: offsetx + size]
            cutout = F.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(torch.cat(cutouts))

# (edit by afiaka87)
#
# Compare `prompt` with 1000 imagenet label transcriptions for its classes.
# `imagenet_classes` list taken from https://github.com/openai/CLIP/notebooks/


def top_imagenet_class(target_text, clip_model=None, device=None, top: int = 16):
    target_text = clip.tokenize(target_text).to(device)
    text_tokenized = clip.tokenize(IMAGENET_CLASSES).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokenized).float()
        target_text_features = clip_model.encode_text(target_text).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    target_text_features /= target_text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * target_text_features @
                  text_features.T).softmax(dim=-1)
    sorted_probs, sorted_classes = text_probs.cpu().topk(top, dim=-1, sorted=True)
    return sorted_classes[0][0]


"""
[Generate an image from a specified text prompt.]
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prompt", type=str, help="the prompt to reward")
    p.add_argument("--prompt_min", type=str, default="", help="the prompt to penalize")
    p.add_argument("--image_size", type=int, default=128,
                   help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", type=str,
                   help="Blend an image with diffusion for n steps")
    p.add_argument('--skip_timesteps', type=int, default=0,
                   help='Number of timesteps to blend image for. CLIP guidance occurs after this.')
    p.add_argument("--num_cutouts", "-cutn", type=int, default=64,
                   help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--prefix", "--output_dir", default="outputs",
                   type=str, help="output directory")
    p.add_argument("--batch_size", "-bs", type=int,
                   default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=int, default=1000,
                   help="Scale for CLIP spherical distance loss. Default value varies depending on image size.")
    p.add_argument("--tv_scale", "-tvs", type=int, default=100,
                   help="Scale for denoising loss. Disabled by default for 64 and 128")
    p.add_argument("--seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-sf", type=int,
                   default=25, help="Save frequency")
    p.add_argument("--device", type=str,
                   help="device to run on .e.g. cuda:0 or cpu")
    p.add_argument("--diffusion_steps", type=int,
                   default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", type=str,
                   default='1000', help="Timestep respacing")
    p.add_argument('--cutout_power', '-cutpow', type=float,
                   default=0.5, help='Cutout size power')
    p.add_argument('--clip_model', type=str, default='ViT-B/32',
                   help='clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]')
    p.add_argument('--class_cond', type=bool, default=True,
                   help='Use class conditional. Required for image sizes other than 256')
    p.add_argument('--custom_class', type=int, default=None,
                   help='Custom class to use for image generation. Should be one of: [0-999]')
    p.add_argument('--clip_class_search', action='store_true',
                   help='Lookup imagenet class with CLIP rather than changing them throughout run. Use `--clip_class_search` on its own to enable. ')
    args = p.parse_args()

    # Initialize
    prompt = args.prompt
    prompt_min = args.prompt_min
    clip_guidance_scale = args.clip_guidance_scale
    init_image = args.init_image
    skip_timesteps = args.skip_timesteps
    image_size = args.image_size
    batch_size = args.batch_size
    seed = args.seed
    save_frequency = args.save_frequency
    cutout_power = args.cutout_power
    num_cutouts = args.num_cutouts
    class_cond = args.class_cond
    diffusion_steps = args.diffusion_steps
    timestep_respacing = args.timestep_respacing
    tv_scale = args.tv_scale
    clip_model_name = args.clip_model
    clip_class_search = args.clip_class_search
    custom_class = args.custom_class
    prefix = args.prefix

    # Disable class randomize from RiversHaveWings if we are passing a class in.
    randomize_class = not clip_class_search

    # clip_class_seaarch and custom_class are mutually exclusive
    assert (clip_class_search is False) or (not custom_class)

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        print(f"Using user-specified device {args.device}.")
        device = torch.device(args.device)

    assert clip_model_name in [
        'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'RN50x16'
    ], 'clip model name should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]'
    assert timestep_respacing in [
        '25', '50', '100', '250', '500', '1000', 'ddim25', 'ddim50', 'ddim100', 'ddim250', 'ddim500', 'ddim1000'
    ], 'timestep_respacing should be one of [25, 50, 100, 250, 500, 1000, ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000]'
    assert image_size in [
        64, 128, 256, 512], 'image size should be one of [64, 128, 256, 512]'

    # Setup
    prompt_as_subdir = prompt
    if len(prompt_min) > 0:
        prompt_as_subdir = f'{prompt_as_subdir}_MIN_{prompt_min}'
    prompt_as_subdir = re.sub(r'[^\w\s]', '', f'{prompt_as_subdir}').replace(' ', '_')[:256]  # Remove non-alphabet characters
    prefix_path = Path(f'{prefix}/{prompt_as_subdir}')

    os.makedirs(prefix_path, exist_ok=True)

    if image_size == 64 and clip_guidance_scale > 500:
        print("CLIP guidance scale and TV scale may be too high for 64x64 image.")
        time.sleep(5)

    if args.class_cond:
        diffusion_path = f'./checkpoints/{image_size}x{image_size}_diffusion.pt'
    else:
        diffusion_path = f'checkpoints/256x256_diffusion_uncond.pt'

    if seed is not None:
        torch.manual_seed(seed)

    # Load diffusion and CLIP models
    gd_model, diffusion = load_guided_diffusion(checkpoint_path=diffusion_path, image_size=image_size,
                                                diffusion_steps=diffusion_steps, timestep_respacing=timestep_respacing, device=device, class_cond=class_cond)
    clip_model = clip.load(clip_model_name, jit=False)[
        0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[
                                     0.26862954, 0.26130258, 0.27577711])

    # Random or CLIP-selected imagenet class.
    model_kwargs = {}
    if clip_class_search:
        imagenet_class = top_imagenet_class(
            prompt, clip_model=clip_model, device=device)
        model_kwargs["y"] = torch.Tensor(
            [imagenet_class]).to(torch.long).to(device)
        print(f"The prompt: '{prompt}'")
        print(
            f"Scored well with the ImageNet class idx: '{imagenet_class.item()}'")
        print(f"This index roughly maps to the label:")
        print(IMAGENET_CLASSES[imagenet_class])
    elif custom_class is not None:
        model_kwargs["y"] = torch.Tensor(
            [custom_class]).to(torch.long).to(device)
    else:
        model_kwargs["y"] = torch.zeros(
            [batch_size], device=device, dtype=torch.long)
        print("Randomizing class as generation occurs.")

    make_cutouts = MakeCutouts(
        clip_size, num_cutouts, cutout_size_power=cutout_power, augment_list=[])

    # Embed text with CLIP model
    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

    # Embed penalty text with CLIP model
    text_min_embed = None
    if len(prompt_min) > 0:
        text_min_embed = clip_model.encode_text(clip.tokenize(prompt_min).to(device)).float()
    
    # (Optional) Load image
    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((image_size, image_size), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    # Customize guided-diffusion model with function that uses CLIP guidance.
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device,
                              dtype=torch.long) * current_timestep
            out = diffusion.p_mean_variance(
                gd_model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            cutout_embeds = clip_model.encode_image( clip_in).float().view([num_cutouts, n, -1])
            max_dists = spherical_dist_loss(cutout_embeds, text_embed.unsqueeze(0))
            min_dists = 0
            if len(prompt_min) > 0:
                min_dists = spherical_dist_loss(cutout_embeds, text_min_embed.unsqueeze(0))
                dists = (0.75 * max_dists) - (0.25 * min_dists) # TODO make these kwargs
            else:
                dists = max_dists
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            return -torch.autograd.grad(loss, x)[0]

    if timestep_respacing.startswith("ddim"):
        diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
    else:
        diffusion_sample_loop = diffusion.p_sample_loop_progressive

    samples = diffusion_sample_loop(
        gd_model,
        (batch_size, 3, image_size, image_size),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        progress=True,
        skip_timesteps=skip_timesteps,
        init_image=init,
        randomize_class=randomize_class,
    )

    print(f"Attempting to generate the caption: '{prompt}'")
    print(f"Penalizing the prompt: '{prompt_min}'")
    print(f"Using initial image: {init_image}")
    print(f"Using {image_size} image size")
    print(f"Using {clip_model_name} as the CLIP model")
    print(f"Using {clip_size} as the CLIP model's visual input resolution")
    print(f"Using {num_cutouts} cutouts.")
    print(f"Using {cutout_power} cutout power")
    print(f"Using {diffusion_steps} diffusion steps.")
    print(f"Using {timestep_respacing} iterations.")
    print(f"Skipping {skip_timesteps} timesteps.")
    print(f"For init image {init_image}")
    print(f"Using {clip_guidance_scale} for text clip guidance scale.")
    print(f"Using {tv_scale} for denoising loss.")
    try:
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for j, image in enumerate(sample["pred_xstart"]):
                    filename = os.path.join(
                        prefix_path, f"{j:04}_iteration_{step:04}.png")
                    pil_image = TF.to_pil_image(
                        image.add(1).div(2).clamp(0, 1))
                    pil_image.save(filename)
                    pil_image.save('current.png')
                    tqdm.write(f"Step {step}, output {j}:")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                f"CUDA OOM error occurred. Lower the batch_size or num_cutouts and try again.")
        else:
            raise e


if __name__ == "__main__":
    main()
