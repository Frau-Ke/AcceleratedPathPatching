torch.cuda.empty_cache()
gc.collect()

device = args.device
if "llama" in args.model:
    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0
elif "gpt2" in args.model:
    # GPT2 Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



for metric in ['IFV', 'WIFV', 'WIFN']:
    args.metric = metric

    
    if "llama" in args.model:
        # Llama Model
        model = get_Llama(args.model, args.cache_dir, device=device, use_4bit=args.use_4bit)
    elif "gpt2" in args.model:
        # GPT2 Model
        model = get_gpt2_adapt("gpt2", "/mnt/lustre/work/eickhoff/esx670/llm_weights")

    torch.cuda.empty_cache()
    device = torch.device(device)
    model.to(device)
    model.eval()

    # run FLAP
    CIRCUIT, scores = prune_flap(args, model, tokenizer, device)
    
    ave_logit_FLAP, performance_FLAP = evaluate_circiut(
        model = model_mean_ablation, 
        CIRCUIT=CIRCUIT,
        dataset=dataset,
        ave_logit_gt=ave_logit_gt
    )

    gain_FLAP = performance_gain(performance_new=performance_FLAP, performance_old=performance_IOI)

    print_statics(
        title=f"*********** FLAP CIRCUIT w. {144 - args.remove_heads} heads*************",
        ave_logit=ave_logit_FLAP, 
        performance_achieved=performance_FLAP,
        CIRCUIT=CIRCUIT, 
        IOI_CIRCUIT=IOI_CIRCUIT,
        performance_gain=gain_FLAP
        )

    heat_map_sparsity(
        scores, 
        IOI_CIRCUIT,
        CIRCUIT, 
        title=f"FLAP CIRCUIT w. {144 - args.remove_heads} heads",
        performance=performance_FLAP)
    
        