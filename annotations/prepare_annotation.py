import json
import argparse

def convert_json_to_jsonl(input_file, output_file):
    """
    Converte un file JSON con una struttura specifica in un file JSONL.
    
    Args:
        input_file (str): Percorso del file JSON di input
        output_file (str): Percorso del file JSONL di output
    """
    # Leggi il file JSON di input
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Apri il file JSONL di output
    with open(output_file, 'w') as f:
        # Itera attraverso le annotazioni
        for annotation in data.get('annotations', []):
            # Estrai i percorsi dei video e li converte in una lista
            video_paths = list(annotation.get('video_paths', {}).values())
            
            # Estrai il livello di competenza
            proficiency_level = annotation.get('proficiency_score', '')
            
            task_id = annotation.get('take_uid', '')
            
            scenario_name = annotation.get('scenario_name', '')
            # Crea un nuovo oggetto nel formato desiderato
            new_entry = {
                'video_paths': video_paths,
                'analysis': '',
                'proficiency_level': proficiency_level,
                'take_uid': task_id,
                'scenario_name': scenario_name
            }
            
            # Scrivi l'oggetto nel file JSONL
            f.write(json.dumps(new_entry) + '\n')
    
    print(f"Conversione completata. Il file è stato salvato come {output_file}")

def main():
    # Configurazione degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Converte un file JSON in formato JSONL.')
    parser.add_argument('--input', '-i', type=str, default='/home/tellong/stockage2/EgoExo4D/annotations/proficiency_demonstrator_train.json', help='Percorso del file JSON di input')
    parser.add_argument('--output', '-o', type=str, default='/home/tellong/Bureau/Code/Code/GazeSkill/SkillFormer/annotations/annotations_train.jsonl', help='Percorso del file JSONL di output')
    
    # Parsing degli argomenti
    args = parser.parse_args()
    
    # Esegui la conversione
    convert_json_to_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()