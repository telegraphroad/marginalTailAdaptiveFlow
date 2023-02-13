for nl in 3 12 8 5
do
    for dim in 2 8
    do
        for marginal in "vanilla" "gTAF"
        do
            for flow in "maf" "nsf"
            do
                python main.py --dim "$dim" --marginals "$marginal" --flow "$flow" --num_layers "$nl" --num_heavy 1 --df 2 --seed 42
            done
        done
    done
done