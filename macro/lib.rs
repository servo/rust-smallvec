use proc_macro::{
    Delimiter,
    TokenStream,
    TokenTree
};

#[proc_macro_attribute]
pub fn syntax(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let mut element = Vec::<TokenTree>::new();
    let mut iterator = item.into_iter().fuse();
    loop {
        match [iterator.next(), iterator.next()] {
            [
                Some(TokenTree::Punct(punctuation)),
                Some(TokenTree::Group(group))
            ] if (punctuation.as_char(), group.delimiter()) == ('#', Delimiter::Bracket) => {
                element.extend([punctuation.into(), group.into()]);
            }
            mut trees => {
                element.extend(
                    attribute
                        .into_iter()
                        .chain(trees.iter_mut().flat_map(Option::take))
                        .chain(iterator)
                );
                return element.into_iter().collect();
            }
        };
    }
}
