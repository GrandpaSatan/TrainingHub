type FieldNoteProps = {
  note: string;
  link: string;
};

export function FieldNote({ note, link }: FieldNoteProps) {
  const href = link.startsWith("#") ? `#/knowledge/${link.slice(1)}` : link;

  return (
    <p className="fieldNote">
      {note} <a href={href}>KB</a>
    </p>
  );
}
